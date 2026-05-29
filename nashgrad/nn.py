
import random
from nashgrad.base import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    # Override this to include the list of parameters.
    def parameters(self):
        return []

class Neuron(Module):
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
    

class Layer(Module):
    # nin = 
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):
    # list of nouts: defines the sizes of all layers in MLP
    # nin in the number of inputs for the whole MLP.
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.nin = nin
        print(f"MLP size list: {sizes}")
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(0, len(sizes)-1)]

    def __call__(self, x):
        assert len(x) == self.nin, f"Input size {x} should be {self.nin}"
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"