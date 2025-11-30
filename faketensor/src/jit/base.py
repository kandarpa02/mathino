from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# Import the EXACT SAME function class used by faketensor autograd
from faketensor.src.base import function, no_record


# ================================================================
# GLOBALS
# ================================================================
_JIT_TRACING: bool = False
_ACTIVE_RECORDER: "JITRecorder | None" = None


# ================================================================
# STATIC OP REPR
# ================================================================
@dataclass
class StaticOp:
    primitive: function             # the autograd primitive wrapper itself
    in_ids: List[int]               # IDs of input objects
    out_id: int                     # ID of output object
    kwargs: Dict[str, Any]          # reserved for future use


# ================================================================
# TRACE RECORDER
# ================================================================
class JITRecorder:
    def __init__(self):
        self.env: Dict[int, int] = {}     # pyobj → stable ID
        self._next: int = 0
        self.ops: List[StaticOp] = []

    def id(self, obj):
        """Assign stable integer ID based on python object identity."""
        k = id(obj)
        if k not in self.env:
            self.env[k] = self._next
            self._next += 1
        return self.env[k]

    def record(self, prim: function, parents, out):
        """Record one primitive execution."""
        in_ids = [self.id(p) for p in parents]
        out_id = self.id(out)
        self.ops.append(StaticOp(prim, in_ids, out_id, {}))


# ================================================================
# TRACE MODE
# ================================================================
@contextmanager
def trace_mode():
    """Enable graph recording for a single forward pass."""
    global _JIT_TRACING, _ACTIVE_RECORDER

    prev = _JIT_TRACING
    _JIT_TRACING = True
    _ACTIVE_RECORDER = JITRecorder()

    try:
        yield
    finally:
        _JIT_TRACING = prev


# ================================================================
# STATIC GRAPH
# ================================================================
class StaticGraph:
    def __init__(self, ops, input_ids, output_id):
        self.ops = ops
        self.input_ids = input_ids
        self.output_id = output_id

    def __call__(self, *args):
        """Replay graph with AUTOGRAD OFF."""
        env = {}

        # Bind runtime inputs
        for a, iid in zip(args, self.input_ids):
            env[iid] = a

        # Replay primitives with autograd completely disabled
        with no_record():
            for op in self.ops:
                ins = [env[i] for i in op.in_ids]
                out = op.primitive(*ins)        # direct eager execution
                env[op.out_id] = out

        return env[self.output_id]


# ================================================================
# FREEZE GRAPH
# ================================================================
def freeze_graph(args, out):
    r = _ACTIVE_RECORDER
    if r is None:
        raise RuntimeError("freeze_graph() called with no active recorder")

    input_ids = [r.id(a) for a in args]
    output_id = r.id(out)
    return StaticGraph(r.ops, input_ids, output_id)


# ================================================================
# INSTALL PRIMITIVE HOOK
# ================================================================
def _install_jit_hook():
    """Monkey-patch function.__call__ exactly once globally."""

    old_call = function.__call__

    def new_call(self, *args):
        # Regular autograd call
        out = old_call(self, *args)

        # If we are tracing → RECORD THIS OP
        if _JIT_TRACING and _ACTIVE_RECORDER is not None:
            parents = getattr(self, "last_parents", ())
            _ACTIVE_RECORDER.record(self, parents, out)

        return out

    function.__call__ = new_call


# Install hook at module import
_install_jit_hook()


