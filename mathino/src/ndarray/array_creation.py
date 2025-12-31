from ...backend import backend as b
from ..array import NDarray
from ...src.DType import DType, normalize_dtype
from typing import Optional

def ones(shape, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().ones(shape, dtype))

def zeros(shape, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().zeros(shape, dtype))

def full(shape, value, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().full(shape, value, dtype))

def ones_like(_data, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().ones_like(_data.np, dtype))

def zeros_like(_data, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().zeros_like(_data.np, dtype))

def full_like(_data, value, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().full_like(_data.np, value, dtype))

def one_hot(labels, num_classes, dtype=None):
    out = zeros(labels.shape + (num_classes,), dtype=dtype)
    valid = (labels >= 0) & (labels < num_classes)
    out[valid, labels[valid]] = 1
    return out