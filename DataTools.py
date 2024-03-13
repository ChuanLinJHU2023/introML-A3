import numpy as np
from DataPredictors import *
import pandas as pd

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


class Model:
    def __init__(self, seq: Sequential, opt: Optimizer, loss: Loss, label_feature):
        self.seq=seq
        self.opt=opt
        self.loss=loss
        self.label_feature=label_feature
        self.label_index=None

    def fit(self, df_train: pd.DataFrame, training_mode):
        assert training_mode=="mini_batch" or training_mode=="incremental" or training_mode=="full"
        try:
            arr_train=np.array(df_train)
        except:
            raise ValueError
        self.label_index=list(df_train.columns).index(self.label_feature)
        batch_size=self.get_batch_size(df_train, training_mode)


    def predict(self):
        pass

    def get_batch_size(self,df_train, training_mode):
        if



