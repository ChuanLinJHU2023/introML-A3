import numpy as np
from DataLayers import *
import pandas as pd


class Loss:
    def __init__(self, input_size=None):
        self.input_size = input_size

    def init_input_size(self, input_size):
        assert self.input_size is None
        self.input_size = input_size

    def value(self, predicted, ground_truth):
        raise NotImplementedError

    def grad(self, predicted, ground_truth):
        raise NotImplementedError

    def input_check(self, predicted, ground_truth):
        assert predicted.shape == (self.input_size, 1)
        assert isinstance(ground_truth, float) or isinstance(ground_truth, int)


class SSE(Loss):
    def value(self, predicted, ground_truth):
        self.input_check(predicted, ground_truth)
        predicted = predicted[0][0]
        value = np.sum(predicted - ground_truth)
        return value

    def grad(self, predicted, ground_truth):
        self.input_check(predicted, ground_truth)
        predicted = predicted[0][0]
        grad = 2 * (predicted - ground_truth)
        grad = np.array([[grad]])
        assert grad.shape == (1, self.input_size)
        return grad


class SoftmaxCrossEntropy(Loss):
    def value(self, predicted, ground_truth):
        self.input_check(predicted, ground_truth)
        probabilities_pred = self.softmax(predicted)
        probabilities_true = self.one_hot_encode(ground_truth)
        return self.cross_entropy(probabilities_pred, probabilities_true)

    def softmax(self, predicted):
        exp = np.exp(predicted)
        exp_sum = np.sum(exp)
        probabilities = exp / exp_sum
        return probabilities

    def cross_entropy(self, probabilities_pred, probabilities_true):
        assert probabilities_pred.shape==probabilities_true.shape==(self.input_size,1)
        return np.sum(np.log2(probabilities_pred + 1e-30) * probabilities_true)

    def one_hot_encode(self, class_k):
        number_of_classes = self.input_size
        shape = (number_of_classes, 1)
        probabilities = np.zeros(shape)
        probabilities[class_k][0] = 1
        return probabilities

    def grad(self, predicted, ground_truth):
        self.input_check(predicted, ground_truth)
        probabilities_pred = self.softmax(predicted)
        probabilities_true = self.one_hot_encode(ground_truth)
        grad = probabilities_pred - probabilities_true
        grad = grad.reshape((1,-1))
        assert grad.shape == (1, self.input_size)
        return grad
