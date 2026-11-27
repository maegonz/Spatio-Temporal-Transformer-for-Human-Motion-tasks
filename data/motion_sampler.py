import random as rd
from torch.utils.data import Sampler
from collections import defaultdict


class MotionSampler(Sampler):
    def __init__(self, frames, batch_size=32, bucket_size=32, shuffle=True, drop_last=False):
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
        self.drop_last = drop_last

        # Group indices by frames
        buckets = defaultdict(list)
        for id, frame in enumerate(frames):
            bucket_id = frame // self.bucket_size
            buckets[bucket_id].append(id)
        self.buckets = [buckets[k] for k in sorted(buckets)]

        if drop_last:
            self._num_batches = sum(len(b) // batch_size for b in self.buckets)
        else:
            self._num_batches = sum(-(-len(b) // batch_size) for b in self.buckets)


    def __len__(self):
        return self._num_batches
    
    def __iter__(self):
        batches, leftovers = [], []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle:
                rd.shuffle(indices)
            full = len(indices) // self.batch_size * self.batch_size
            batches += [indices[i:i + self.batch_size] for i in range(0, full, self.batch_size)]
            leftovers += indices[full:]
            
        for i in range(0, len(indices), self.batch_size):
            batch = leftovers[i:i + self.batch_size]
            if not self.drop_last or len(batch) == self.batch_size:
                continue
            batches.append(batch)

        if self.shuffle:
            rd.shuffle(batches)
            
        yield from batches