import numpy as np
from typing import List
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.input=None
        self.output=None
        self.params=None
        self.grads=None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def param(self, parameter_name=None):
        raise NotImplementedError

    def grad(self, parameter_name=None):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size, W:np.ndarray = None, b:np.ndarray = None):
        super().__init__(input_size, output_size)
        W=np.random.randn(output_size,input_size) if W is None else W
        b=np.random.randn(output_size,1) if b is None else b
        W_grad=0
        b_grad=0
        self.params=[W,b]
        self.grads=[W_grad,b_grad]

    def forward(self, input):
        assert input.shape==(self.input_size,1)
        self.input=input
        W, b = self.params
        self.output= W @ input + b
        return self.output

    def backward(self, output_gradient):
        assert output_gradient.shape==(1,self.output_size)
        W, b = self.params
        input_gradient=output_gradient @ W
        W_grad, b_grad = self.grads
        W_grad += output_gradient.T @ self.input.T
        b_grad += output_gradient.T
        assert W_grad.shape==W.shape
        assert b_grad.shape==b.shape
        self.grads=[W_grad, b_grad]
        return input_gradient

    def param(self, parameter_name=None):
        assert parameter_name
        W = self.params[0]
        b = self.params[1]
        if parameter_name=="W":
            return W
        if parameter_name=="b":
            return b
        return None

    def grad(self, parameter_name=None):
        assert parameter_name
        W_grad=self.grads[0]
        b_grad=self.grads[1]
        if parameter_name=="W":
            return W_grad
        if parameter_name=="b":
            return b_grad
        return None


class Sigmoid(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        assert input_size==output_size

    def forward(self, input):
        assert input.shape==(self.input_size,1)
        self.input=input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient):
        assert output_gradient.shape==(1,self.output_size)
        input_gradient=self.output*(1-self.output)*output_gradient
        return input_gradient


class Sequential:
    def __init__(self, layers: List[Layer]):
        assert isinstance(layers, list)
        self.layers = layers
        self.input_size = layers[0].input_size
        self.output_size = layers[1].input_size

    def forward(self, input):
        assert input.shape == (self.input_size, 1)
        intermediate = input
        for layer in self.layers:
            intermediate = layer.forward(intermediate)
        output = intermediate
        return output

    def backward(self, output_gradient):
        assert output_gradient.shape == (1, self.output_size)
        intermediate_gradient = output_gradient
        for layer in reversed(self.layers):
            intermediate_gradient = layer.backward(intermediate_gradient)
        input_gradient = intermediate_gradient
        return input_gradient


