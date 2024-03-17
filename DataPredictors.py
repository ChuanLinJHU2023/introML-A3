import numpy as np
from DataOptimizers import *
from DataLossers import *
import pandas as pd


class DNN:

    def __init__(self, seq: Sequential, opt: Optimizer, loss: Loss, label_feature, batch_size=None, n_epochs=None,
                 clear_after_pred=False):
        self.seq: Sequential = seq
        self.opt: Optimizer = opt
        self.loss: Loss = loss
        self.label_feature = label_feature
        self.label_feature_index = None
        self.whether_encode = isinstance(loss, SoftmaxCrossEntropy)
        self.whether_decode = isinstance(loss, SoftmaxCrossEntropy)
        self.batch_Size = batch_size
        self.n_epochs = n_epochs
        self.clear_after_pred = clear_after_pred

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
        if self.clear_after_pred:
            self.seq.clear_params()
        if show_detail:
            print("the predictions are as follows:")
            print(ys)
        return ys



class AutoEncoder:

    def __init__(self, n_shared_layes,  label_feature,
                 seq_enc: Sequential, seq_pred: Sequential,
                 opt_enc: Optimizer, opt_pred: Optimizer,
                 loss_enc: Loss, loss_pred: Loss,
                 batch_size_enc, batch_size_pred,
                 n_epochs_enc, n_epochs_pred,
                 clear_after_pred=False):
        self.n_shared_layers = n_shared_layes
        # two sequentials
        self.seq_enc: Sequential = seq_enc
        self.seq_pred: Sequential = seq_pred
        # two optimizers
        self.opt_enc: Optimizer = opt_enc
        self.opt_pred: Optimizer = opt_pred
        # two lossers
        self.loss_enc: Loss = loss_enc
        self.loss_pred: Loss = loss_pred
        # two batch sizes
        self.batch_size_enc = batch_size_enc
        self.batch_size_pred = batch_size_pred
        # two epoch numbers
        self.n_epochs_enc = n_epochs_enc
        self.n_epochs_pred = n_epochs_pred
        # the rest
        self.label_feature = label_feature
        self.label_feature_index = None
        self.whether_encode = isinstance(loss_pred, SoftmaxCrossEntropy)
        self.whether_decode = isinstance(loss_pred, SoftmaxCrossEntropy)
        self.clear_after_pred = clear_after_pred

    def fit_for_encoder(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
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
                    predicted = self.seq_pred.forward(x, need_reshape=True)
                    ground_truth = y
                    loss = self.loss_pred.value(predicted, ground_truth, need_reshape=True,
                                                need_encode=self.whether_encode)
                    epoch_loss += loss
                    grad = self.loss_pred.grad()
                    self.seq_pred.backward(grad)
                self.opt_pred.step(self.seq_pred)
            print("Epoch {} for Predictor: Loss {}".format(epoch_i, epoch_loss))

    def fit_for_predictor(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
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
                    predicted = self.seq_pred.forward(x, need_reshape=True)
                    ground_truth = y
                    loss = self.loss_pred.value(predicted, ground_truth, need_reshape=True,
                                                need_encode=self.whether_encode)
                    epoch_loss += loss
                    grad = self.loss_pred.grad()
                    self.seq_pred.backward(grad)
                self.opt_pred.step(self.seq_pred)
            print("Epoch {} for Predictor: Loss {}".format(epoch_i, epoch_loss))

    def predict(self, df_test: pd.DataFrame, show_detail=False):
        arr_test = np.array(df_test)
        datapoint_number = arr_test.shape[0]
        xs = np.delete(arr_test, self.label_feature_index, axis=1)
        ys = np.zeros(datapoint_number)
        for dp_i in range(datapoint_number):
            x = xs[dp_i]
            predicted = self.seq_pred.forward(x, need_reshape=True, need_decode=self.whether_decode)
            ys[dp_i] = predicted
        if self.clear_after_pred:
            self.seq_enc.clear_params()
            self.seq_pred.clear_params()
        if show_detail:
            print("the predictions are as follows:")
            print(ys)
        return ys

    def check_encoder_and_predictor(self):
        for i in range(self.n_shared_layers):
            assert self.seq_enc.layers[i] is self.seq_pred.layers[i]

