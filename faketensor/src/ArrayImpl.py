import numpy as np
from typing import NamedTuple
from ..src.autograd.tape_recorder import Node
from typing import Callable, Optional, Any

def opr(*args, fn):
    data = fn(*[arg.d for arg in args])
    return NDarray(data)

def make_op(fw:Callable, grad_fn:Callable|Any=None, name:str|Any=None, method:bool=True):
    """
    fw: forward function, takes raw np.ndarray arguments
    grad_fn: backward function, takes grad, *args and returns tuple of grads
    name: name of the method to attach
    method: if True, attach to NDarray, else return a pure function
    """

    def broadcast_backward(grad, x_shape, y_shape):
        """
        Sum-reduce grad to match x_shape after broadcasting
        """
        # Add singleton dimensions to grad if necessary
        while len(grad.shape) > len(x_shape):
            grad = grad.sum(axis=0)
        # Sum over broadcasted axes
        for i, (sx, sg) in enumerate(zip(x_shape, grad.shape)):
            if sx == 1 and sg != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def wrapper(*args):
        # Extract raw data
        data_args = [arg.d if isinstance(arg, NDarray) else arg for arg in args]
        out_data = fw(*data_args)
        out = NDarray(out_data)

        if grad_fn is not None:
            # Keep only NDarray parents
            parents = tuple(arg for arg in args if isinstance(arg, NDarray))

            def backward(grad):
                grads = grad_fn(grad, *args)
                # Handle broadcasting
                final_grads = []
                for g, parent in zip(grads, parents):
                    g_bc = broadcast_backward(g, parent.d.shape, out.d.shape)
                    final_grads.append(g_bc)
                return tuple(final_grads)

            out.tensornode = Node(out, parents=parents, backward=backward)

        return out

    if method and name is not None:
        setattr(NDarray, name, wrapper)
    else:
        return wrapper

class NDarray:
    def __init__(self, data):
        self.d = np.asarray(data)
        self.tensornode = None
        self.grad = np.array(0.0)
        self.shape = self.d.shape

    def __repr__(self):
        return self.d.__repr__()

    def __str__(self):
        return self.d.__str__()

    def __add__(self, other):
        out = opr(self, other, fn=lambda a, b: a+b)
        out.tensornode = Node(out, parents=(self, other), backward=lambda grad: (grad, grad))
        return out
    
    def __radd__(self, other):
        return self + other

    # def __mul__(self, other):
    #     out = opr(self, other, fn=lambda a, b: a*b)
    #     out.tensornode = Node(out, parents=(self, other), backward=lambda grad: (grad*other, grad*self))
    #     return out
    
    # def __rmul__(self, other):
    #     return self * other
    
fw_div = lambda a, b: a / b
grad_div = lambda g, a, b: (g / b, -g * a / (b**2))

make_op(fw_div, grad_fn=grad_div, name="__truediv__")
make_op(fw_div, grad_fn=grad_div, name="__rtruediv__")

fw_mul = lambda a, b: a * b
grad_mul = lambda g, a, b: (g * b, g * a)

make_op(fw_mul, grad_fn=grad_mul, name="__mul__")
make_op(fw_mul, grad_fn=grad_mul, name="__rmul__")