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
        self.label_feature_index = None

    def fit(self, df_train: pd.DataFrame, batch_size, n_epochs):
        arr_train = np.array(df_train)
        self.label_feature_index = list(df_train.columns).index(self.label_feature)
        datapoint_number = arr_train.shape[0]
        xs = np.delete(arr_train, self.label_feature_index, axis=1)
        ys = arr_train[:, self.label_feature_index]
        for epoch_i in range(n_epochs):
            epoch_loss=0
            for batch_i in range(datapoint_number // batch_size):
                start = batch_i * batch_size
                end = min((batch_i + 1) * batch_size, datapoint_number)
                for dp_i in range(start, end):
                    x = xs[dp_i].reshape((-1, 1))
                    y = ys[dp_i]
                    predicted = self.seq.forward(x)
                    ground_truth = y
                    loss = self.loss.value(predicted, ground_truth, need_reshape=True)
                    epoch_loss += loss
                    grad = self.loss.grad()
                    self.seq.backward(grad)
                self.opt.step(self.seq)
                print("Epoch {}: Loss {}".format(epoch_i, epoch_loss))

    def predict(self, df_test: pd.DataFrame):
        arr_test = np.array(df_test)
        datapoint_number = arr_test.shape[0]
        xs = np.delete(arr_test, self.label_feature_index, axis=1)
        ys = np.zeros(datapoint_number)
        start = 0
        end = datapoint_number
        for dp_i in range(start, end):
            x = xs[dp_i].reshape((-1, 1))
            predicted = self.seq.forward(x)
            ys[dp_i] = predicted
        return ys
