import numpy as np
import pandas as pd

from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *
from DataValidators import *

df=data_loader("Datasets/breast-cancer-wisconsin.data",missing_value_symbol="?",idCol=0)
print("Cancer Dataset")
print("The original dataset is:")
print(df,"\n\n")
df=data_imputer(df,[1,2,3,4,5,6,7,8,9])
df=data_standardizer(df,[1,2,3,4,5,6,7,8,9])
df=data_label_rename(df,10)
print("The processed dataset is:")
print(df,"\n\n")
label_feature = 10
type = "c"
metric = "MSE" if type == "r" else "01Loss"
print("the class labels are:", np.unique(df[10]))


# For the following models
input_size = df.shape[1] - 1
output_size = 4
lr = 0.01
n_epochs = 500
batch_size = 50
print("The input size is {}".format(input_size))
print("The output size is {}".format(output_size))
print("The learning rate is {}".format(lr))
print("The epoch number is {}".format(n_epochs))
print("The batch size is {}\n\n".format(batch_size))
loss = SoftmaxCrossEntropy(output_size=output_size) if type=="c" else SSE(output_size=output_size)
opt = GradientDecsent(lr=lr)


# The first model is the Null model
null_model = Null(type, label_feature=label_feature)


# Linear Model
linear_layer1_for_m1 = Linear(input_size, output_size)
layers_for_m1 = [linear_layer1_for_m1]
seq_for_m1 = Sequential(layers_for_m1)
linear_model = DNN(seq_for_m1, opt, loss, label_feature=label_feature, batch_size=batch_size,
                   n_epochs=n_epochs, clear_after_pred=True, show_detials=False)


# 2-hidden-layer FNN Model
hidden_size1 = 20
hidden_size2 = 10
print("The hidden size for FNN are {} and {}".format(hidden_size1, hidden_size2))
linear_layer1_for_m2 = Linear(input_size, hidden_size1)
sigmoid_layer1_for_m2 = Sigmoid(hidden_size1, hidden_size1)
linear_layer2_for_m2 = Linear(hidden_size1, hidden_size2)
sigmoid_layer2_for_m2 = Sigmoid(hidden_size2, hidden_size2)
linear_layer3_for_m2 = Linear(hidden_size2, output_size)
layers_for_m2 = [linear_layer1_for_m2, sigmoid_layer1_for_m2, linear_layer2_for_m2, sigmoid_layer2_for_m2,
                 linear_layer3_for_m2]
seq_for_m2 = Sequential(layers_for_m2)
FNN_model = DNN(seq_for_m1, opt, loss, label_feature=label_feature, batch_size=batch_size,
                n_epochs=n_epochs, clear_after_pred=True, show_detials=False)


# Autoencoder Model (1 hidden layer for auto network and 2 hidden layers for predictor network)
hidden_size1 = 4
hidden_size2 = 10
print("The hidden size for Autoencoder are {} and {}".format(hidden_size1, hidden_size2))
assert hidden_size1<=input_size
linear_layer1_for_m3 = Linear(input_size, hidden_size1)
sigmoid_layer1_for_m3 = Sigmoid(hidden_size1, hidden_size1)
linear_layer2_for_m3 = Linear(hidden_size1, hidden_size2)
sigmoid_layer2_for_m3 = Sigmoid(hidden_size2, hidden_size2)
linear_layer3_for_m3 = Linear(hidden_size2, output_size)
layers_for_m3 = [linear_layer1_for_m3, sigmoid_layer1_for_m3, linear_layer2_for_m3, sigmoid_layer2_for_m3,
                 linear_layer3_for_m3]
seq_for_m3 = Sequential(layers_for_m3)

linear_layer4_for_m3 = Linear(hidden_size1, input_size)
layers_for_m3_enc = [linear_layer1_for_m3, sigmoid_layer1_for_m3, linear_layer4_for_m3]
seq_for_m3_enc =Sequential(layers_for_m3_enc)
AE_model = AutoEncoder(seq_for_m3_enc, seq_for_m3, opt, loss,
                    label_feature=label_feature, batch_size=batch_size, n_epochs=n_epochs, clear_after_pred=False)

predictors = [null_model, linear_model, FNN_model, AE_model]
validator = KByTwoValidatorMultiple(predictors, k=5, evaluation_metric=metric)
validator.validate(df, show_detail=True, show_more_detail=True)
