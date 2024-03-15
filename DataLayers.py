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

    def forward(self, input, need_reshape=False):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def param(self, parameter_name=None):
        raise NotImplementedError

    def grad(self, parameter_name=None):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size, W:np.ndarray = None, b:np.ndarray = None, examplary=False):
        super().__init__(input_size, output_size)
        W=np.random.randn(output_size,input_size) if W is None else W
        b=np.random.randn(output_size,1) if b is None else b
        assert W.shape==(output_size,input_size)
        assert b.shape==(output_size,1)
        if examplary:
            W=np.arange(output_size*input_size).reshape((output_size,input_size))
            b = np.arange(output_size).reshape((output_size,1))
        W_grad=np.zeros((output_size,input_size))
        b_grad=np.zeros((output_size,1))
        self.params=[W,b]
        self.grads=[W_grad,b_grad]

    def forward(self, input, need_reshape=False):
        if need_reshape:
            input=np.array(input).reshape(-1,1)
        assert input.shape==(self.input_size,1)
        self.input=input
        W, b = self.params
        self.output= W @ input + b
        assert self.output.shape==(self.output_size,1)
        return self.output

    def backward(self, output_gradient):
        assert output_gradient.shape==(self.output_size,1)
        W, b = self.params
        input_gradient= W.T @ output_gradient
        W_grad, b_grad = self.grads
        W_grad += output_gradient @ self.input.T
        b_grad += output_gradient
        self.grads=[W_grad, b_grad]
        assert W_grad.shape==W.shape
        assert b_grad.shape==b.shape
        assert input_gradient.shape==(self.input_size,1)
        return input_gradient

    def param(self, parameter_name=None):
        assert parameter_name
        W, b = self.params
        if parameter_name=="W":
            return W
        if parameter_name=="b":
            return b
        return None

    def grad(self, parameter_name=None):
        assert parameter_name
        W_grad, b_grad=self.grads
        if parameter_name=="W":
            return W_grad
        if parameter_name=="b":
            return b_grad
        return None


class Sigmoid(Layer):
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)

    def forward(self, input, need_reshape=False):
        assert input.shape==(self.input_size,1)
        self.input=input
        self.output = 1 / (1 + np.exp(-input))
        assert self.output.shape == (self.output_size, 1)
        return self.output

    def backward(self, output_gradient):
        assert output_gradient.shape==(self.output_size,1)
        input_gradient=self.output * (1-self.output)* output_gradient
        assert input_gradient.shape==(self.input_size,1)
        return input_gradient


class Sequential:
    def __init__(self, layers: List[Layer]):
        assert isinstance(layers, list)
        self.layers = layers
        self.input_size = layers[0].input_size
        self.output_size = layers[-1].output_size

    def forward(self, input, need_reshape=False, need_decode=False):
        if need_reshape:
            input=np.array(input).reshape(-1,1)
        assert input.shape == (self.input_size, 1)
        intermediate = input
        for layer in self.layers:
            intermediate = layer.forward(intermediate)
        output = intermediate
        assert output.shape == (self.output_size,1)
        if need_decode:
            output=self.one_hot_decode(output)
        return output

    def backward(self, output_gradient):
        assert output_gradient.shape == (self.output_size, 1)
        intermediate_gradient = output_gradient
        for layer in reversed(self.layers):
            intermediate_gradient = layer.backward(intermediate_gradient)
        input_gradient = intermediate_gradient
        assert input_gradient.shape == (self.input_size, 1)
        return input_gradient

    def one_hot_decode(self, prediction):
        assert prediction.shape == (self.output_size, 1)
        return np.argmax(prediction)


