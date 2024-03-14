import numpy as np
from DataLayers import *
import pandas as pd
from typing import *
class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

    def step(self, seq: Sequential):
        raise NotImplementedError


class GradientDecsent(Optimizer):
    def step(self, seq_or_lay: Union[Sequential, Layer]):
        if isinstance(seq_or_lay, Sequential):
            seq=seq_or_lay
            for layer in seq.layers:
                self.step(layer)
        else:
            layer=seq_or_lay
            for i in range(len(layer.params)):
                assert layer.params[i].shape == layer.grads[i].shape
                layer.params[i] -= layer.grads[i] * self.learning_rate
                layer.grads[i][:]=0


