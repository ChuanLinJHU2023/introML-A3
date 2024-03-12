import numpy as np
class Layer:
    def __init__(self, input_size, output_size):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def param(self, parameter_name=None):
        raise NotImplementedError

    def grad(self, parameter_name=None):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size, W=None, b=None):
        self.input_size=input_size
        self.output_size=output_size
        self.input=None
        self.output=None
        self.W=np.random.randn(output_size,input_size) if W is None else W
        self.b=np.random.randn(output_size,1) if b is None else b
        self.W_grad=None
        self.b_grad=None

    def forward(self, input):
        assert isinstance(input,np.ndarray)
        assert input.shape==(self.input_size,1)
        self.input=input
        self.output=self.W @ input + self.b
        return self.output

    def backward(self, output_gradient):
        assert isinstance(output_gradient,np.ndarray)
        assert output_gradient.shape==(1,self.output_size)
        input_gradient=output_gradient@self.W
        self.W_grad=output_gradient@self.input
        self.b_grad=output_gradient
        assert self.W_grad.shape==self.W.shape
        assert self.b_grad.shape==self.b.shape
        return input_gradient

    def param(self, parameter_name):
        assert parameter_name
        if parameter_name=="W":
            return self.W
        if parameter_name=="b":
            return self.b
        return None

    def grad(self, parameter_name=None):
        assert parameter_name
        if parameter_name=="W":
            return self.W_grad
        if parameter_name=="b":
            return self.b_grad
        return None


class Sigmoid(Layer):
    def __init__(self, input_size, output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.input=None
        self.output=None

    def forward(self, input):
        assert isinstance(input,np.ndarray)
        assert input.shape==(self.input_size,1)
        self.input=input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient):
        input_gradient=self.output*(1-self.output)*output_gradient
        return input_gradient

