# from ..backend.backend import xp
# lib = xp()
import numpy as np
import math
import random
from ..src.tree_util import map
from ..src.ndarray.base import array
from typing import List

def len_error(*args):
    first_len = len(args[0])
    val = all(len(arg) == first_len for arg in args)
    if not val:
        raise ValueError(f"input arrays must have the same length.")

class ARRAYLOADER:
    def __len__(self): pass
    def __getitem__(self, idx) -> List: pass

class ArrayLoader(ARRAYLOADER):
    def __init__(
        self,
        *args,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        split: tuple | None = None,
        part: int = 0,
    ):
        len_error(args)

        # from .datasets import dataset
        # if len(args) == 1 and isinstance(args[0], dataset):
        #     args = args[0].arrays

        self.ddict = dict(enumerate(args))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # ---- base indices ----
        total_samples = len(args[0])
        indices = np.arange(total_samples)

        # ---- split logic ----
        if split is not None:
            if not isinstance(split, tuple):
                raise TypeError("split must be a tuple")

            if any(s <= 0 for s in split):
                raise ValueError("split values must be positive")

            total = sum(split)
            if total > 100:
                raise ValueError("split percentages must sum to <= 100")

            if len(split) == 1:
                split = (split[0], 100 - split[0])
            elif total < 100:
                split = split + (100 - total,)

            sizes = [int(total_samples * s / 100) for s in split]
            sizes[-1] = total_samples - sum(sizes[:-1])

            bounds = np.cumsum([0] + sizes)

            if part >= len(sizes):
                raise ValueError("part index out of range")

            indices = indices[bounds[part]:bounds[part + 1]]

        # ---- shuffle inside split ----
        if shuffle:
            indices = np.random.permutation(indices)

        self.indices = indices
        self.num_samples = len(indices)

        # ---- batch count ----
        if drop_last:
            self.num_batches = self.num_samples // batch_size
        else:
            self.num_batches = math.ceil(self.num_samples / batch_size)

    def __len__(self):
        """Number of batches"""
        return self.num_batches

    def __getitem__(self, batch_idx):
        if batch_idx >= self.num_batches:
            raise IndexError("Batch index out of range")

        start = batch_idx * self.batch_size
        end = start + self.batch_size

        idx = self.indices[start:end]
        ret = list(map(lambda x: array(x[idx]), self.ddict).values())
        return ret[0] if len(ret) == 1 else ret


def array_loader(
        *args,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        split: tuple | None = None,
        part: int = 0,
    ):
    
    loader = ArrayLoader(*args, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, split=split, part=part)
    return loader