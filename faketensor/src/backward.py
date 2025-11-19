from typing import List, Callable, Any, Tuple
import numpy as np
from .base import TAPE_STACK, tape
from .array import NDarray

def grad(fun: Callable, argnum: int = 0) -> Callable:
    """
    Creates a gradient function similar to JAX's grad.
    
    Args:
        fun: Function to differentiate
        argnum: Which argument to differentiate with respect to
    
    Returns:
        Function that computes gradient of fun wrt argnum-th argument
    """
    def gradfun(*args):
        # Run forward pass to build computation graph
        # args = list(arg.np for arg in args)
        with tape():
            output = fun(*args)
        
        # Get the tape from the context
        tape_records = TAPE_STACK[-1] if TAPE_STACK else []
        
        # Initialize gradients
        grads = {id(output): np.ones_like(output)}
        
        # Backward pass: traverse tape in reverse
        for node in reversed(tape_records):
            # Get gradient for this node's output
            g_out = grads.get(id(node.out))
            if g_out is None:
                continue
                
            # Compute gradients for parents
            parent_grads = node.grad_fn(g_out)
            
            # Accumulate gradients for parents
            for parent, parent_grad in zip(node.parents, parent_grads):
                parent_id = id(parent)
                if parent_id in grads:
                    grads[parent_id] = grads[parent_id] + parent_grad
                else:
                    grads[parent_id] = parent_grad
        
        # Return gradient for the requested argument
        target_arg = args[argnum] if argnum < len(args) else None
        if target_arg is not None:
            return grads.get(id(target_arg), np.zeros_like(target_arg))
        else:
            # Handle keyword arguments (simplified)
            return None
    
    return gradfun

def value_and_grad(fun: Callable, argnum: int = 0) -> Callable:
    """
    Returns both value and gradient, like JAX's value_and_grad.
    """
    def value_gradfun(*args, **kwargs):
        with tape():
            value = fun(*args, **kwargs)
        grad_val = grad(fun, argnum)(*args, **kwargs)
        return value, grad_val
    
    return value_gradfun
