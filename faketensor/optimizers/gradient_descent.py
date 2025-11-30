from .base import Optimizer
from ..neural_nets.base import Cell

class GradientDescent(Optimizer):
    def __init__(self, model:Cell, lr=0.1):
        super().__init__(model)
        self.lr = lr  # use argument, you hardcoded 0.1 earlier

    def update_rule(self, grads):
        new_params = [
            p - self.lr * g
            for p, g in zip(self.model.parameters(), grads)
        ]
        return new_params

class SGD(Optimizer):
    def __init__(self, model: Cell, lr=0.1, nestrov=True):
        super().__init__(model)