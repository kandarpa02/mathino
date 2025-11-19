import numpy as np
from typing import Callable

def broadcast_backward(grad: np.ndarray, x_shape: tuple) -> np.ndarray:
    # Remove leading dims added by broadcasting
    while len(grad.shape) > len(x_shape):
        grad = grad.sum(axis=0)

    # Reduce along broadcasted axes
    for i, (sx, sg) in enumerate(zip(x_shape, grad.shape)):
        if sx == 1 and sg != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
