import numpy as np
from DataOptimizers import *
from DataLossers import *
import pandas as pd


class Null:
    """
    NullPredictor implements the Null predictor for classification or regression tasks.

    predictor_type (str): Indicates the type of predictor, either "c" for classification or "r" for regression.
    label_feature (str): The name of the label feature in the dataset.
    """

    predictor_type = None
    label_feature = None
    prediction_value = None

    def __init__(self, predictor_type=None, label_feature=None):
        """initialize the Null predictor"""
        assert predictor_type == "c" or predictor_type == "r"  # classification or regression
        self.predictor_type = predictor_type
        self.label_feature = label_feature

    def fit(self, df_train, relearn_VDM=False):
        """let Null predictor fit the training dataset"""
        assert self.label_feature in df_train.columns, "your 'labelName parameter may be wrong"
        if self.predictor_type == "c":
            self.prediction_value = df_train[self.label_feature].mode()[0]
        elif self.predictor_type == "r":
            self.prediction_value = df_train[self.label_feature].mean()
        return 0

    def predict(self, df_test):
        """let Null predictor predict the test data set"""
        predictedValues = pd.Series([self.prediction_value] * df_test.shape[0], index=df_test.index)
        return predictedValues

    def predict_for_single_datapoint(self, s_datapoint):
        """let Null predictor predict a single datapoint"""
        return self.prediction_value

    def get_label_feature(self):
        """get the label feature"""
        return self.label_feature


class DNN:

    def __init__(self, seq: Sequential, opt: Optimizer, loss: Loss, label_feature, batch_size=None, n_epochs=None,
                 clear_after_pred=False, show_detials=False):
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
        self.show_details = show_detials

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
            if self.show_details:
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

    def get_label_feature(self):
        return self.label_feature


class AutoEncoder:

    def __init__(self, seq_enc: Sequential, seq: Sequential, opt: Optimizer, loss: Loss, label_feature,
                 batch_size=None, n_epochs=None, clear_after_pred=False, n_shared_layers=2, show_details=False):
        self.seq_enc: Sequential = seq_enc
        self.seq: Sequential = seq
        self.opt: Optimizer = opt
        self.loss_enc = SSE(self.seq_enc.output_size)
        self.loss: Loss = loss
        self.label_feature = label_feature
        self.label_feature_index = None
        self.whether_encode = isinstance(loss, SoftmaxCrossEntropy)
        self.whether_decode = isinstance(loss, SoftmaxCrossEntropy)
        self.batch_Size = batch_size
        self.n_epochs = n_epochs
        self.clear_after_pred = clear_after_pred
        self.n_shared_layers = n_shared_layers
        self.check_encoder_and_predictor()
        self.show_details = show_details

    def fit(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
        self.fit_for_encoder(df_train, batch_size, n_epochs)
        self.fit_for_predictor(df_train, batch_size, n_epochs)
        return 0

    def fit_for_encoder(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
        batch_size = self.batch_Size if batch_size is None else batch_size
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        arr_train = np.array(df_train)
        self.label_feature_index = list(df_train.columns).index(self.label_feature)
        datapoint_number = arr_train.shape[0]
        xs = np.delete(arr_train, self.label_feature_index, axis=1)
        for epoch_i in range(n_epochs):
            epoch_loss = 0
            for batch_i in range(datapoint_number // batch_size):
                start = batch_i * batch_size
                end = (batch_i + 1) * batch_size
                for dp_i in range(start, end):
                    x = xs[dp_i]
                    predicted = self.seq_enc.forward(x, need_reshape=True)
                    ground_truth = x
                    loss = self.loss_enc.value(predicted, ground_truth, need_reshape=True,
                                           need_encode=self.whether_encode)
                    epoch_loss += loss
                    grad = self.loss_enc.grad()
                    self.seq_enc.backward(grad)
                self.opt.step(self.seq_enc)
            if self.show_details:
                print("Epoch {} for Encoder: Loss {}".format(epoch_i, epoch_loss))

    def fit_for_predictor(self, df_train: pd.DataFrame, batch_size=None, n_epochs=None):
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
            if self.show_details:
                print("Epoch {} for Predictor: Loss {}".format(epoch_i, epoch_loss))

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
            self.seq_enc.clear_params()
            self.seq.clear_params()
        if show_detail:
            print("the predictions are as follows:")
            print(ys)
        return ys

    def check_encoder_and_predictor(self):
        assert self.seq_enc.output_size == self.seq_enc.input_size
        for i in range(self.n_shared_layers):
            assert self.seq_enc.layers[i] is self.seq.layers[i]

    def get_label_feature(self):
        return self.label_feature