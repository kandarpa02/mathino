from ..src.array import NDarray

class Variable(NDarray):
    def __init__(self, data, dtype=None):
        super().__init__(data, dtype)
        self.train = True

    @property
    def trainable(self):
        return self.train
    
    def freeze(self):
        self.train = False

    def unfreeze(self):
        self.train = True

