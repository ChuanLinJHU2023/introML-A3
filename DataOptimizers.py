import numpy as np
from DataLayers import *
import pandas as pd
from typing import *
class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr
        self.show_weight_update_process = False

    def step(self, seq: Sequential):
        raise NotImplementedError


class GradientDecsent(Optimizer):
    def step(self, seq_or_lay: Union[Sequential, Layer]):
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


