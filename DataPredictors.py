import numpy as np
from DataOptimizers import *
from DataLossers import *
import pandas as pd


class DNN:

    def __init__(self, seq: Sequential, opt: Optimizer, loss: Loss, label_feature, batch_size=None, n_epochs=None):
        self.seq: Sequential = seq
        self.opt: Optimizer = opt
        self.loss: Loss = loss
        self.label_feature = label_feature
        self.label_feature_index = None
        self.whether_encode = isinstance(loss, SoftmaxCrossEntropy)
        self.whether_decode = isinstance(loss, SoftmaxCrossEntropy)
        self.batch_Size = batch_size
        self.n_epochs = n_epochs

    def fit(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
        batch_size = self.batch_Size if batch_size is None else batch_size
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        arr_train = np.array(df_train)
        self.label_feature_index = list(df_train.columns).index(self.label_feature)
        datapoint_number = arr_train.shape[0]
        xs = np.delete(arr_train, self.label_feature_index, axis=1)
        ys = arr_train[:, self.label_feature_index]
        for epoch_i in range(n_epochs):
            epoch_loss = 0
            for batch_i in range(datapoint_number // batch_size):
                start = batch_i * batch_size
                end = (batch_i + 1) * batch_size
                for dp_i in range(start, end):
                    x = xs[dp_i]
                    y = ys[dp_i]
                    predicted = self.seq.forward(x, need_reshape=True)
                    ground_truth = y
                    loss = self.loss.value(predicted, ground_truth, need_reshape=True,
                                           need_encode=self.whether_encode)
                    epoch_loss += loss
                    grad = self.loss.grad()
                    self.seq.backward(grad)
                self.opt.step(self.seq)
            print("Epoch {}: Loss {}".format(epoch_i, epoch_loss))

    def predict(self, df_test: pd.DataFrame, show_detail=False):
        arr_test = np.array(df_test)
        datapoint_number = arr_test.shape[0]
        xs = np.delete(arr_test, self.label_feature_index, axis=1)
        ys = np.zeros(datapoint_number)
        for dp_i in range(datapoint_number):
            x = xs[dp_i]
            predicted = self.seq.forward(x, need_reshape=True, need_decode=self.whether_decode)
            ys[dp_i] = predicted
        if show_detail:
            print("the predictions are as follows:")
            print(ys)
        return ys
