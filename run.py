import faketensor as ft 
from faketensor import ndarray as nd
import numpy as np

class Linear(ft.Cell):
    def __init__(self, _in, out):
        super().__init__()
        self.w = ft.Variable(np.random.rand(_in, out))
        self.b = ft.Variable(np.zeros(out))


    def call(self, x):
        return ft.matmul(x, self.w) + self.b

class Model(ft.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(5, 3)
        self.f2 = Linear(3, 1)

    def call(self, x):
        return self.f2(self.f1(x))
    

model = Model()

a = nd.array(np.random.rand(4, 5))
params = list(model.parameters())

print(model(a))
print(params)