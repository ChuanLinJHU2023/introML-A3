import numpy as np
from DataPredictors import *

class Loss:
    def __init__(self, predicted, ground_truth):
        self.predicted=predicted
        self.ground_truth=ground_truth

    def value(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class SSE(Loss):
    def __init__(self, predicted, ground_truth):
        assert predicted.shape==ground_truth.shape
        super().__init__(predicted, ground_truth)

    def value(self):
        value=np.sum(self.predicted - self.ground_truth)
        return value

    def grad(self):
        return 2*(self.predicted-self.ground_truth)


class Optimizer:
    def __init__(self,lr):
        self.learning_rate=lr

    def step(self, seq: Sequential):
        raise NotImplementedError


class GradientDecsent(Optimizer):
    def step(self, seq: Sequential):
        for layer in seq.layers:
            for i in range(len(layer.params)):
                assert layer.params[i].shape==layer.grads[i].shape
                shape=layer.params[i].shape
                layer.params[i]+=layer.grads[i]
                layer.grads[i]=np.zeros(shape)


# class Model:
#     def __init__(self,):





