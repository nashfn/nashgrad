
import random
from nashgrad.base import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #print(f"called with {x}")
        # wi*xi+b and then tanh() of that
        t = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return t.tanh()

    def __repr__(self):
        return f"Neuron({self.w}, {self.b})"

    def parameters(self):
        return [self.b] + self.w