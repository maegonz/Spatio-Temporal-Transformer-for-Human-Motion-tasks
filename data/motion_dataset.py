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

# directories containing text descriptions and motion files
text_dir = './data/texts/'  # directory containing all .txt description files
motion_dir = './data/motions/'  # directory containing all .npy motion files

class MotionDataset(Dataset):
    def __init__(self,
                 file: str = "train",
                 text_dir: str = text_dir,
                 motion_dir: str = motion_dir,
                 tokenizer_name: str = 't5-small',
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        Params
        -------
        file : str
            File containing the list of data samples to be used.
            "val", "train" or "test". Defaults to "train".
        """

        assert file in ["train", "val", "test"], "file argument must be one of 'train', 'val', or 'test'"
        self.file = file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.mean_std = os.path.exists("train_mean_std.npz")

        # path of the text descripions and motions directory
        # .../text/
        # .../motions/
        self.text_dir = text_dir
        self.motion_dir = motion_dir

        with open(os.path.join('./data/', f'{file}.txt'), 'r') as f:
            self.files_name = set(f.read().splitlines())

        # get sorted list of all text files and motion files
        all_text = sorted(glob(os.path.join(text_dir, '*.txt')))
        all_motion = sorted(glob(os.path.join(motion_dir, '*.npy')))

        self.text_files = sorted([i for i in all_text if os.path.basename(i).split('.')[0] in self.files_name])
        self.motion_files = sorted([i for i in all_motion if os.path.basename(i).split('.')[0] in self.files_name])
        
        self.motion_frames = []
        for m in self.motion_files:
            # mmap_mode='r' allows us to read the shape of the npy file without loading the entire file into memory
            shape = np.load(m, mmap_mode='r').shape
            self.motion_frames.append(shape[0])

        # Normalisation
        if file == "train":
            if mean is None or std is None or not self.mean_std:
                self.compute_mean_std()
            elif (mean is None or std is None) and self.mean_std:
                npzfile = np.load("train_mean_std.npz")
                self.mean = npzfile['x']
                self.std = npzfile['y']
            else:
                self.mean = mean
                self.std = std
        else:
            # Load mean and std from train dataset
            assert mean is not None and std is not None, "Mean and std must be provided for validation and test datasets."
            self.mean = mean
            self.std = std

    def compute_mean_std(self):
        sum_, sum_sq, count = 0, 0, 0
        for m in self.motion_files:
            motion = np.load(m)  # [T, J, 3]
            sum_ += motion.sum(axis=0)        # [J, 3]
            sum_sq += (motion ** 2).sum(axis=0) # [J, 3]
            count += motion.shape[0]            # T

        self.mean = sum_ / count
        self.std = np.sqrt(sum_sq / count - (self.mean ** 2)) + 1e-8  # avoid division by zero
        
        if not self.mean_std:
            np.savez("train_mean_std.npz", x=self.mean, y=self.std)

    def __len__(self):
        return len(self.files_name)
    
    def __getitem__(self, idx):
        # read npy motion file
        motion = np.load(self.motion_files[idx], mmap_mode='r')  # [T, J, 3]
        motion = (motion - self.mean) / (self.std)               # normalize the motion data
        motion = torch.from_numpy(motion.copy()).float()         # convert to torch tensor
    
        if self.file != "test":
            # get the corresponding description for the associated motion
            with open(self.text_files[idx], 'r', encoding='utf-8') as f:
                descriptions = [
                    caption.split('#')[0].capitalize() for caption in f.readlines()
                ]
            
            #TODO: deal with multiple captions
            text = rd.choice(descriptions)

            # Tokenize the text description
            tokens = self.tokenizer(text, max_length=64, truncation=True, return_tensors=None)

            return {
                "motion": motion,
                "captions": text,
                "input_ids": tokens["input_ids"],  # shape: (1, seq_len)
            }
        else:
            return {
                "motion": motion,
            }