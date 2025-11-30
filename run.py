import faketensor as ft 
from faketensor import ndarray as nd
import numpy as np

class Linear(ft.Cell):
    def __init__(self, _in, out):
        super().__init__()
        np.random.seed(0)
        self.w = ft.Variable(np.random.rand(_in, out), name='weight')
        self.b = ft.Variable(np.zeros(out), name='bias')


    def call(self, x):
        return ft.matmul(x, self.w) + self.b

class Model(ft.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(3, 1)

    def call(self, x):
        return self.f1(x)
    

model = Model()
optimizer = ft.GradientDescent(model, lr=0.2)


np.random.seed(0)
a = nd.array(np.random.rand(20, 3))
b = nd.array(np.random.rand(20))

def loss_f(model, x, y):
    pred = model(x)
    loss = ft.mean((pred - y) ** 2)
    return loss

# def steps(epochs):
#     for e in range(epochs+1):
out, grads = ft.value_and_grad(lambda model:loss_f(model, a, b))(model)
optimizer.update(grads)

state = optimizer.get_state()

optimizer.load_state(state)

# print(optimizer)

@ft.jit
def f(x, y):
    print("Tracing")
    return (x * x) / y

a = nd.array(3.)
b = nd.array(4.)

print(f(a, b))
print(f(a, b))