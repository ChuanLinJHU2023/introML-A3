# DataOptimizers.py implements the gradient descent optimizer.
from DataLayers import *
from typing import *


class Optimizer:
    """
    Base class for all optimizers.

    Attributes:
        - lr (float): The learning rate used by the optimizer.
        - show_weight_update_process (bool): Whether to show the weight update process during training.
    """


    def __init__(self, lr):
        self.learning_rate = lr
        self.show_weight_update_process = False

    def step(self, seq: Sequential):
        """
        Performs a single optimization step on the given sequence or layer.

        Parameters:
            seq (Union[Sequential, Layer]): The sequence or layer to perform optimization on.

        """
        raise NotImplementedError


class GradientDecsent(Optimizer):
    """
    Class implementing the gradient descent optimizer.
    """
    def step(self, seq_or_lay: Union[Sequential, Layer]):
        """
        Performs a single optimization step on the given sequence or layer.

        Args:
            seq_or_lay (Union[Sequential, Layer]): The sequence or layer to perform optimization on.
        """
        if isinstance(seq_or_lay, Sequential):
            seq=seq_or_lay
            for layer in seq.layers:
                if self.show_weight_update_process:
                    print("in layer {}".format(seq.layers.index(layer)))
                    print("this is {} layer".format(type(layer).__name__))
                    params = layer.params if layer.params else list()
                    grads = layer.grads if layer.grads else list()
                    print("before update, the params are")
                    for param in params:
                        print(param)
                    print("before update, the grads are")
                    for grad in grads:
                        print(grad)
                self.step(layer)
                if self.show_weight_update_process:
                    print("after update, the params are")
                    params = layer.params if layer.params else list()
                    for param in params:
                        print(param)
                    print("\n\n")
            self.show_weight_update_process = False
        else:
            layer=seq_or_lay
            if layer.params is not None:
                for i in range(len(layer.params)):
                    assert layer.params[i].shape == layer.grads[i].shape
                    layer.params[i] -= layer.grads[i] * self.learning_rate
                    layer.grads[i][:]=0


