# DataLayers.py implements the Linear Layer, the Sigmoid Layer, and other layers

import numpy as np
from typing import List
show_forward_propagation_process = False

class Layer:
    """Base class for neural network layers, including the Linear Layer and the Sigmoid Layer.
    However, a Fully-connected layer is implemented as a tuple of a linear layer and a sigmoid layer"""

    def __init__(self, input_size, output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.input=None
        self.output=None
        self.params=None
        self.grads=None

    def forward(self, input, need_reshape=False):
        """
        Compute the forward pass of the layer.

       Args:
           input (np.array): Input tensor. It should be a column vector
           need_reshape (bool): Whether reshaping is needed for the output.

       Returns:
           np.array: Output tensor after applying the layer's transformation.
       """
        raise NotImplementedError

    def backward(self, output_gradient):
        """
        Compute the backward pass of the layer.

       Args:
           output_gradient (np.array): Gradient of the loss with respect to the output. It should be a column vector

       Returns:
           np.array: Gradient of the loss with respect to the input.
        """
        raise NotImplementedError

    def param(self, parameter_name=None):
        """"Get the parameters of a layer"""
        raise NotImplementedError

    def grad(self, parameter_name=None):
        """"Get the gradients to parameters of a layer"""
        raise NotImplementedError

    def clear_params(self):
        """Clear the learned parameters of a layer"""
        raise NotImplementedError


class Linear(Layer):
    """
    Linear layer implementation. It should do linear transformation on the input.

    Args:
        input_size (int): The size of the input
        output_size (int): The size of the output
        W : Weight matrix. Defaults to None, in which case it is initialized randomly.
        b : Bias vector. Defaults to None, in which case it is initialized randomly.
        exemplary : If True, initializes weights and biases in an exemplary manner. Defaults to False.

    Attributes:
        params (list): List containing weights and biases.
        grads (list): List containing gradients of weights and biases.
    """

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
        """
            Performs forward pass.

               Args:
                   input (np.array): Input data.
                   need_reshape (bool): Whether to reshape the input. Defaults to False.

               Returns:
                   np.array: Output of the forward pass.
               """
        if need_reshape:
            input=np.array(input).reshape(-1,1)
        assert input.shape==(self.input_size,1)
        self.input=input
        W, b = self.params
        self.output= W @ input + b
        assert self.output.shape==(self.output_size,1)
        return self.output

    def backward(self, output_gradient):
        """
        Performs backward pass.

        Args:
            output_gradient (np.array): Gradient of the output.

        Returns:
            np.array: Gradient of the input.
        """
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
        """
        Retrieves parameters.

        Args:
            parameter_name (str): Name of the parameter to retrieve.

        Returns:
            np.array: Retrieved parameter.
        """
        assert parameter_name
        W, b = self.params
        if parameter_name=="W":
            return W
        if parameter_name=="b":
            return b
        return None

    def grad(self, parameter_name=None):
        """
        Retrieves gradients.

        Args:
            parameter_name (str): Name of the parameter whose gradient is going to retrieve.

        Returns:
            np.array: Retrieved gradient.
        """
        assert parameter_name
        W_grad, b_grad=self.grads
        if parameter_name=="W":
            return W_grad
        if parameter_name=="b":
            return b_grad
        return None

    def clear_params(self):
        """Reinitializes parameters randomly."""
        W, b = self.params
        W_shape = W.shape
        b_shape = b.shape
        self.params[0] = np.random.randn(*W_shape)
        self.params[1] = np.random.randn(*b_shape)


class Sigmoid(Layer):
    """
    Sigmoid layer implementation.

    Args:
        input_size (int): The size of the input
        output_size (int): The size of the output
    """
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)

    def forward(self, input, need_reshape=False):
        """
        Performs forward pass.

        Args:
            input (np.array): Input data
            need_reshape (bool): Whether to reshape the input into a column vector. Defaults to False.

        Returns:
            np.array: Output of the forward pass.
        """
        assert input.shape==(self.input_size,1)
        self.input=input
        self.output = 1 / (1 + np.exp(-input))
        assert self.output.shape == (self.output_size, 1)
        return self.output

    def backward(self, output_gradient):
        """
        Performs backward pass.

        Args:
            output_gradient (np.array): Gradient of the output.

        Returns:
            np.ndarray: Gradient of the input.
        """
        assert output_gradient.shape==(self.output_size,1)
        input_gradient=self.output * (1-self.output)* output_gradient
        assert input_gradient.shape==(self.input_size,1)
        return input_gradient

    def clear_params(self):
        pass


class Sequential:
    """
    Sequential model that stacks layers sequentially.

    Args:
        layers (List[Layer]): List of layer objects.

    Attributes:
        layers: List of layer objects.
        input_size: Size of the input
        output_size: Size of the output
        show_forward_propagation_process: Flag to control whether to print the forward propagation process.
    """

    def __init__(self, layers: List[Layer]):
        """
        Initializes the Sequential model.

        Args:
            layers (List[Layer]): List of layer objects.
        """
        self.layers = layers
        self.input_size = layers[0].input_size
        self.output_size = layers[-1].output_size
        self.check_layers()
        self.show_forward_propagation_process = False

    def forward(self, input, need_reshape=False, need_decode=False):
        """
        Performs forward pass.

        Args:
            input (np.array): Input data.
            need_reshape (bool): Whether to reshape the input. Defaults to False.
            need_decode (bool): Whether to decode the output. Defaults to False.

        Returns:
            np.ndarray: Output of the forward pass.
        """
        if need_reshape:
            input=np.array(input).reshape(-1,1)
        assert input.shape == (self.input_size, 1)
        intermediate = input
        for layer in self.layers:
            if self.show_forward_propagation_process:
                print("in layer {}".format(self.layers.index(layer)))
                print("this is {} layer".format(type(layer).__name__))
                print("the input is:")
                print(intermediate)
                print("the output is:")
                print(layer.forward(intermediate))
                print("The layer parameters are :")
                if layer.params:
                    for param in layer.params:
                        print(param)
                print("\n")
            intermediate = layer.forward(intermediate)
        output = intermediate
        assert output.shape == (self.output_size,1)
        if need_decode:
            output=self.one_hot_decode(output)
        return output

    def backward(self, output_gradient):
        """
        Performs backward pass.

        Args:
            output_gradient (np.array): Gradient of the output.

        Returns:
            np.array: Gradient of the input.
        """
        assert output_gradient.shape == (self.output_size, 1)
        intermediate_gradient = output_gradient
        for layer in reversed(self.layers):
            intermediate_gradient = layer.backward(intermediate_gradient)
        input_gradient = intermediate_gradient
        assert input_gradient.shape == (self.input_size, 1)
        return input_gradient

    def one_hot_decode(self, prediction):
        """
        Decodes one-hot encoded prediction.

        Args:
            prediction (np.array): One-hot encoded prediction.

        Returns:
            int: Decoded class label.
        """
        assert prediction.shape == (self.output_size, 1)
        return np.argmax(prediction)

    def show_params(self):
        """Prints parameters and outputs of each layer."""
        for i,layer in enumerate(self.layers):
            print("This is the params for layer {}".format(i))
            print(layer.params)
            print("This is the output for layer {}".format(i))
            print(layer.output)

    def clear_params(self):
        """Clears parameters of all layers."""
        for layer in self.layers:
            layer.clear_params()

    def check_layers(self):
        """Checks if the output size of each layer matches the input size of the next layer."""
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            assert layer.output_size == next_layer.input_size




