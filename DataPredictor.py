import numpy as np
from DataLayers import *
from DataOptimizers import *
from DataLossers import *
import pandas as pd

class DNN_Model:
    def __init__(self, seq: Sequential, opt: Optimizer, loss: Loss, label_feature):
        self.seq = seq
        self.opt = opt
        self.loss = loss
        self.label_feature = label_feature
        self.label_index = None

    def fit(self, df_train: pd.DataFrame, batch_size, n_epochs):
        arr_train = np.array(df_train)
        self.label_index = list(df_train.columns).index(self.label_feature)
        datapoint_number = arr_train.shape[0]
        feature_number = arr_train.shape[1] - 1
        xs = np.delete(arr_train, self.label_index, axis=1)
        ys = arr_train[:, self.label_index]
        for epoch_i in range(n_epochs):
            for batch_i in range(arr_train.shape[0] // batch_size):
                start = batch_i * batch_size
                end = min((batch_i + 1) * batch_size, datapoint_number)
                epoch_loss = 0
                for dp_i in range(start, end):
                    x = xs[dp_i].reshape((feature_number, 1))
                    y = ys[dp_i]
                    predicted = self.seq.forward(x)
                    ground_truth = y
                    loss = self.loss.value(predicted, ground_truth)
                    grad = self.loss.grad(predicted, ground_truth)
                    self.seq.backward(grad)
                    epoch_loss += loss
                self.opt.step(self.seq)
                print("Epoch {}: Loss {}".format(epoch_i, epoch_loss))

    def predict(self, df_test: pd.DataFrame):
        arr_test = np.array(df_test)
        datapoint_number = arr_test.shape[0]
        feature_number = arr_test.shape[1] - 1
        xs = np.delete(arr_test, self.label_index, axis=1)
        ys = np.zeros(datapoint_number)
        start = 0
        end = datapoint_number
        for dp_i in range(start, end):
            x = xs[dp_i].reshape((feature_number, 1))
            predited = self.seq.forward(x)
            ys[dp_i] = predited
        return ys
