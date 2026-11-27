import numpy as np
import os
import random as rd
import torch
from glob import glob
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
from collections import defaultdict

FPS = 20            # HumanML3D (KIT-ML: 12.5)
MIN_LEN, MAX_KEEP, CROP_LEN = 40, 200, 196

class MotionDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 text_dir: str = './data/texts/',
                 motion_dir: str = './data/motions/',
                 tokenizer_name: str = 't5-small',
                 max_tokens: int = 64,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 mean_std_path: str = "./data/train_mean_std.npz"):
        """
        Params
        -------
        split : str, optional
            Dataset split to load: ``"train"``, ``"val"``, or ``"test"``.
            Defaults to ``"train"``.
        text_dir : str, optional
            Directory containing the text description files.
        motion_dir : str, optional
            Directory containing the NumPy motion files.
        tokenizer_name : str, optional
            Hugging Face tokenizer name or path used for captions.
        mean : np.ndarray, optional
            Motion feature means used for normalization.
        std : np.ndarray, optional
            Motion feature standard deviations used for normalization.
        mean_std_path : str, optional
            Path to the ``.npz`` file containing training means and standard
            deviations. If absent, statistics are computed for the training
            split; validation and test splits require this file or supplied
            statistics.

        Attributes
        ----------
        split : str
            Name of the dataset split (``"train"``, ``"val"``, or ``"test"``).
        train : bool
            Whether the dataset is the training split.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer used to encode caption text.
        samples : list[dict]
            Loaded motion segments, captions, and tokenized captions.
        mean : np.ndarray
            Motion feature means used for normalization.
        std : np.ndarray
            Motion feature standard deviations used for normalization.
        """

        assert split in ["train", "val", "test"], "file argument must be one of 'train', 'val', or 'test'"
        self.split, self.train = split, split == "train"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # Read the motion identifiers assigned to this split
        with open(os.path.join('./data/', f'{split}.txt'), 'r') as f:
            names = set(f.read().splitlines())

        # Group captions by their annotated time range for each motion
        self.samples, skipped = [], 0
        for name in names:
            m_path = os.path.join(motion_dir, f"{name}.npy")
            t_path = os.path.join(text_dir, f"{name}.txt")
            if not (os.path.exists(m_path) and os.path.exists(t_path)):
                skipped += 1
                continue
            motion = np.load(m_path).astype(np.float32)           # (T, J, 3)
            if not np.isfinite(motion).all():
                skipped += 1
                continue

            groups = defaultdict(list)                            # (start, end) -> captions
            with open(t_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("#")
                    cap = parts[0].strip()

                    # if cap empty = get the next one
                    if not cap:
                        continue

                    s = _as_float(parts[2]) if len(parts) > 2 else 0.0
                    e = _as_float(parts[3]) if len(parts) > 3 else 0.0
                    groups[(s, e)].append(cap)

            for (s, e), caps in groups.items():
                full = (s == 0.0 and e == 0.0)
                seg = motion if full else motion[int(s * FPS):int(e * FPS)]
                if not (MIN_LEN <= len(seg) < MAX_KEEP):
                    continue
                self.samples.append({
                    "name": name if full else f"{name}_{s:.2f}_{e:.2f}",
                    "motion": seg,
                    "captions": caps,
                    # Tokenize the text description
                    "input_ids": self.tokenizer(caps, max_length=max_tokens,
                                                truncation=True)["input_ids"],
                })

        print(f"{split} {len(self.samples)} samples, {skipped} motions skipped")

        # Normalisation
        if mean is not None or std is not None :
            self.mean, self.std = mean, std
        elif os.path.exists(mean_std_path):
            npzfile = np.load(mean_std_path)
            self.mean, self.std = npzfile['x'], npzfile['y']
        elif self.train:
            # Compute and save mean, std
            self.mean, self.std = self._compute_mean_std()
            np.savez(mean_std_path, x=self.mean, y=self.std)
        else:
            raise ValueError("Validation and test set need train mean/std (pass them as argument or create the stats file)")

    def _compute_mean_std(self):
        # Accumulate feature sums across frames to compute normalization statistics
        sum_, sum_sq, count = 0, 0, 0

        for sample in self.samples:
            motion = sample["motion"].astype(np.float64)  # [T, J, 3]
            sum_ += motion.sum(axis=0)                    # [J, 3]
            sum_sq += (motion ** 2).sum(axis=0)           # [J, 3]
            count += motion.shape[0]                      # T

        mean = sum_ / count
        std = np.sqrt(sum_sq / count - (self.mean ** 2)) + 1e-8  # avoid division by zero
        return mean.astype(np.float32), std.astype(np.float32)

    @property
    def frames(self):                     # for MotionSampler
        return [min(len(s["motion"]), CROP_LEN) for s in self.samples]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # read npy motion file
        sample = self.samples[idx]                               # [T, J, 3]
        motion = sample["motion"]

        # Crop long motions to a fixed length; randomize training crops only
        if len(motion) > CROP_LEN:
            start = rd.randint(0, len(motion) - CROP_LEN) if self.train else 0
            motion = motion[start:start + CROP_LEN]

        motion = (motion - self.mean) / (self.std)               # normalize the motion data
        motion = torch.from_numpy(motion).float()         # convert to torch tensor

        k = rd.randrange(len(sample["captions"])) if self.train else 0
    
        return {
            "motion": motion,
            "input_ids": sample["input_ids"][k],  # shape: (1, seq_len)
            # "captions": sample["captions"][k],
            "references": sample["captions"],  # all captions for val & test
            "name": sample["name"]
        }

def _as_float(s: str):
    try:
        a = float(s)
        return 0.0 if np.isnan(a) else a
    except ValueError:
        return 0.0