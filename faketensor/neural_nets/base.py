from typing import Any
from .parameters import Variable

class Cell:
    def __init__(self):
        self.local_params: list[Variable] = []

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Variable):
            self.local_params.append(value)


    def parameters(self):
        for p in self.local_params:
            yield p
        for v in self.__dict__.values():
            if isinstance(v, Cell):
                yield from v.parameters()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError
