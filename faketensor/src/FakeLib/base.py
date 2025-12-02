from dataclasses import dataclass
from typing import Any, Callable, List, Tuple
import numpy as np

@dataclass
class Tracer:
    out: Any
    parents: Tuple
    op: Callable

class Symbol:
    def __init__(self, shape, dtype, **kwargs):
        self.shape = shape
        self.dtype = dtype
        self.params = kwargs

    def __repr__(self) -> str:
        return f"Symbol({self.shape}, {self.dtype})"

TRACED: List[Tracer] = []

def make_func(fun):
    def runner(*args):
        global TRACED
        out, traced = fun(*args)
        TRACED.append(traced)
        return out
    return runner

@make_func
def add(a, b):
    out = a + b
    _a, _b = Symbol(a.shape, a.dtype), Symbol(b.shape, b.dtype)
    _out = Symbol(out.shape, out.dtype)
    return out, Tracer(_out, parents=(_a, _b), op=lambda x, y: x + y)

@make_func
def mul(a, b):
    out = a * b
    _a, _b = Symbol(a.shape, a.dtype), Symbol(b.shape, b.dtype)
    _out = Symbol(out.shape, out.dtype)
    return out, Tracer(_out, parents=(_a, _b), op=lambda x, y: x * y)


class StaticGraph:
    def __init__(self, fun: Callable):
        self.fun = fun
        self.graph = None  # List[Tracer]

    def __call__(self, *args):
        global TRACED

        if self.graph is None:
            TRACED.clear()
            out = self.fun(*args)
            self.graph = TRACED.copy()
            return out
        else:
            # ---------------------------------
            # Next calls → run static graph
            # ---------------------------------
            return self.run_static(*args)

    def run_static(self, *inputs):
        env = []

        # Fill environment with input tensors
        for x in inputs:
            env.append(x)

        env_i = 0

        # Sequential execution (already topologically sorted)
        for tracer in self.graph:
            p = tracer.parents

            # each parent corresponds to an earlier environment entry
            x = env[env_i]
            y = env[env_i + 1]
            env_i += 2

            out = tracer.op(x, y)
            env.append(out)

        return env[-1]
