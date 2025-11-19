from typing import List, Callable, Protocol, Union
from contextlib import contextmanager

from dataclasses import dataclass
import numpy as np
from .utils import broadcast_backward

from ._typing import arraytype as _ar
arraytype = Union[_ar, np.ndarray]

TAPE_STACK = []

def active_tape():
    return TAPE_STACK[-1] if TAPE_STACK else None

@contextmanager
def tape():
    TAPE_STACK.append([])  
    try:
        yield
    finally:
        pass     

@dataclass
class Node:
    out: np.ndarray
    parents: tuple[np.ndarray]
    grad_fn: Callable

def function_register(fun):
    def inner(*args):
        out, parents, grad_fn = fun(*args)
        t = active_tape()
        if t is not None:
            t.append(Node(out, parents, grad_fn))
        return out
    return inner

def add(x:arraytype, y:arraytype):
    @function_register
    def _fun(x, y):
        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, x.shape)
            return g1, g2
        return np.add(x, y), (x, y), grad_fn
    
    return _fun(x, y)

def mul(x:arraytype, y:arraytype):
    @function_register
    def _fun(x, y):
        def grad_fn(g):
            g1 = broadcast_backward(mul(g, y), x.shape)
            g2 = broadcast_backward(mul(g, x), x.shape)
            return g1, g2
        return np.multiply(x, y), (x, y), grad_fn
    
    return _fun(x, y)
    