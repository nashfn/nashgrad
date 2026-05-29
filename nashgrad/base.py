import math
import numpy


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
    def __repr__(self):
        return f"Value({self.label}={self.data})"
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        # out._backward should propagate its grad and update the grad for self and other.
        # essentially when invoked it should:
        # self.grad = out.grad (since its a + op)
        # other.grad = out.grad.
        # += Multivariate calculus: accumulation of gradients.
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        # += Multivariate calculus: accumulation of gradients.
        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), f'exp({self.data}')
        #print(f"exp called {self}")
        def _backward():
            #print(f"self={self}.grad = {self.grad}, out = {out}")
            self.grad += out.grad * out.data
            #print(f"self={self}.grad = {self.grad}, out.grad = {out.grad}, out.data={out.data}")
        out._backward = _backward
        return out


    def __truediv__(self, other):
        return self * (other**-1)
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        # += Multivariate calculus: accumulation of gradients.
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1 )
        out = Value(t, _children = (self, ), _op='tanh')
        # += Multivariate calculus: accumulation of gradients.
        def _backward():
            # how do we know its the correct value t ? the forward pass will set this method using closure to
            # capture the right value of t.
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out
    def backward(self):
        visited = set()
        topo_order = []
        def build_topo(visited, v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(visited, child)
                topo_order.append(v)
        self.grad = 1.0
        build_topo(visited, self)
        for n in reversed(topo_order):
            n._backward()

    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
