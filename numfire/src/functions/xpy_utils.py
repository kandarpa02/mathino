from ...backend.backend import xp
from xpy.utils import shift_device_

def get_dev(*arrays):
    """
    Determine the device of the given arrays.

    Priority:
        - If any array is a CuPy array -> 'cuda'
        - Otherwise, if any array is a NumPy array -> 'cpu'
        - Otherwise -> None

    Supports:
        - Raw np.ndarray / cp.ndarray
        - NDarray wrappers (via __backend_buffer__)
    """
    import numpy as np
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        cp = None
        has_cupy = False

    # First, check for GPU arrays
    if has_cupy:
        for arr in arrays:
            buf = getattr(arr, "__backend_buffer__", arr)
            if isinstance(buf, cp.ndarray):
                return "cuda"

    # Then, check for CPU arrays
    for arr in arrays:
        buf = getattr(arr, "__backend_buffer__", arr)
        if isinstance(buf, np.ndarray):
            return "cpu"

    for arr in arrays:
        if isinstance(arr, (int, float, bool, list)):
            return "cpu" if xp().__name__ == 'numpy' else "cuda"

def module(type):
    if type == "cuda":
        try:
            import cupy as cp
            return cp
        except ImportError:
            import numpy as np
            return np

    if type == "cpu":
        import numpy as np
        return np

    raise TypeError(f"Unknown backend type: {type}")


from .._typing import Array

def device_shift(x:Array, device:str):
    """Shift to preferred device: 'cpu' or 'cuda'. """
    from ..array import as_nd
    return as_nd(shift_device_(x.__backend_buffer__, device=device))

