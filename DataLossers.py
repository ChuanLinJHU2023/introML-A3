import numpy as np
from DataLayers import *
import pandas as pd


class Loss:
    def __init__(self, input_size=None):
        self.input_size=input_size

    def init_input_size(self,input_size):
        assert self.input_size is None
        self.input_size=input_size

    def value(self, predicted, ground_truth):
        raise NotImplementedError

    def grad(self, predicted, ground_truth):
        raise NotImplementedError


class SSE(Loss):
    def value(self, predicted, ground_truth):
        assert predicted.shape==(self.input_size,1)
        assert isinstance(ground_truth,float) or isinstance(ground_truth,int)
        predicted=predicted[0][0]
        value = np.sum(predicted - ground_truth)
        return value

    def grad(self, predicted, ground_truth):
        assert predicted.shape == (self.input_size, 1)
        assert isinstance(ground_truth, float) or isinstance(ground_truth, int)
        predicted = predicted[0][0]
        grad = 2 * (predicted - ground_truth)
        grad = np.array([[grad]])
        assert grad.shape==(1,self.input_size)
        return grad