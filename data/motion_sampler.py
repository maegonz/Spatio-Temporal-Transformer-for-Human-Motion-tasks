import random as rd
from torch.utils.data import Sampler
from collections import defaultdict


class MotionSampler(Sampler):
    def __init__(self, frames, batch_size=32, bucket_size=32, shuffle=True):
        """
        Sampler that groups sequences of similar frames into the same batch.

        Parameters
        ----------
        frames : list of int
            List containing the frames of each sequence in the dataset.
        batch_size : int
            Number of samples per batch. Defaults to 32.
        shuffle : bool, optional
            Whether to shuffle the batches, by default True.
        """
        self.frames = frames
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        # Group indices by frames
        buckets = defaultdict(list)
        for id, frame in enumerate(frames):
            bucket_id = frame // self.bucket_size
            buckets[bucket_id].append(id)
        
        # Create batches
        self.batches = []
        for bucket in buckets.values():
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                self.batches.append(batch)
        
        if self.shuffle:
            rd.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        yield from self.batches