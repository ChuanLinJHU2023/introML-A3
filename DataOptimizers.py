import numpy as np
from DataLayers import *
import pandas as pd

class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

    def step(self, seq: Sequential):
        raise NotImplementedError


class GradientDecsent(Optimizer):
    def step(self, seq: Sequential):
        for layer in seq.layers:
            for i in range(len(layer.params)):
                assert layer.params[i].shape == layer.grads[i].shape
                layer.params[i] -= layer.grads[i] * self.learning_rate
                layer.grads[i][:]=0

