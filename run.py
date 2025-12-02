import faketensor as ft 
import numpy as np
from faketensor import ndarray as nd

class Linear(ft.nn.Cell):
    def __init__(self, _in, out):
        super().__init__(name='linear')
        self._in = _in
        self.out = out
        np.random.seed(0)
        self.weights = ft.Variable(np.random.rand(_in, out))
        self.bias = ft.Variable(np.zeros(out))
    
    def __repr__(self):
        return f"Linear({self._in}, {self.out})"


    def call(self, x):
        return ft.matmul(x, self.weights) + self.bias

class Enc(ft.nn.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(3, 5)
        self.f2 = Linear(5, 2)

    def call(self, x):
        return self.f2(self.f1(x))
    
class Dec(ft.nn.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(2, 1)
    
    def call(self, x):
        return self.f1(x)

class Model(ft.nn.Cell):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.encoder = Enc()
        self.decoder = Dec()

    def call(self, x):
         return self.decoder(self.encoder(x))

model = Model()
optimizer = ft.optimizers.SGD(model, lr=0.2)

# print(model)

np.random.seed(0)
a = nd.array(np.random.rand(20, 3))
b = nd.array(np.random.rand(20))

def loss_f(model, x, y):
    pred = model(x)
    loss = ft.mean((pred - y) ** 2)
    return loss

x = ft.Variable(4.)

g = ft.grad(lambda x: x*x)

print(g(x))
print(ft.grad(g)(x))

