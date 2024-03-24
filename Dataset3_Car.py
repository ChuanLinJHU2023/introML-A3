import pandas as pd

from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *
from DataValidators import *

df: pd.DataFrame = data_loader("Datasets/car.data")
print("Car Dataset")
print("The original dataset is:")
print(df, "\n\n")
df = data_encode(df, 0, "n", ["low", "med", "high", "vhigh"])
df = data_encode(df, 1, "n", ["low", "med", "high", "vhigh"])
df = data_encode(df, 2, "n", ["2", "3", "4", "5more"])
df = data_encode(df, 3, "n", ["2", "4", "more"])
df = data_encode(df, 4, "n", ["small", "med", "big"])
df = data_encode(df, 5, "n", ["low", "med", "high"])
df = data_encode(df, 6, "o", ["unacc", "acc", "good", "vgood"])
features=list(df.columns)[:-1]
df=data_standardizer(df,features)
print("The processed dataset is:")
print(df, "\n\n")
label_feature = 6
type = "c"
metric = "MSE" if type == "r" else "01Loss"

# The first model is the Null model
null_model = Null(type, label_feature=label_feature)


# For the following models
input_size = df.shape[1] - 1
output_size = 4
lr = 0.01
n_epochs = 1000
batch_size = 50
print("The input size is {}".format(input_size))
print("The output size is {}".format(output_size))
print("The learning rate is {}".format(lr))
print("The epoch number is {}".format(n_epochs))
print("The batch size is {}\n\n".format(batch_size))
loss = SoftmaxCrossEntropy(output_size=output_size)
opt = GradientDecsent(lr=lr)


# Linear Model
linear_layer1_for_m1 = Linear(input_size, output_size)
layers_for_m1 = [linear_layer1_for_m1]
seq_for_m1 = Sequential(layers_for_m1)
linear_model = DNN(seq_for_m1, opt, loss, label_feature=label_feature, batch_size=batch_size,
                   n_epochs=n_epochs, clear_after_pred=True, show_detials=False)


# 2-hidden-layer FNN Model
hidden_size1 = 20
hidden_size2 = 10
linear_layer1_for_m2 = Linear(input_size, hidden_size1)
sigmoid_layer1_for_m2 = Linear(hidden_size1, hidden_size1)
linear_layer2_for_m2 = Linear(hidden_size1, hidden_size2)
sigmoid_layer2_for_m2 = Linear(hidden_size2, hidden_size2)
linear_layer3_for_m2 = Linear(hidden_size2, output_size)
layers_for_m2 = [linear_layer1_for_m2, sigmoid_layer1_for_m2, linear_layer2_for_m2, sigmoid_layer2_for_m2,
                 linear_layer3_for_m2]
seq_for_m2 = Sequential(layers_for_m2)
FNN_model = DNN(seq_for_m1, opt, loss, label_feature=label_feature, batch_size=batch_size,
                n_epochs=n_epochs, clear_after_pred=True, show_detials=False)


# Autoencoder Model (1 hidden layer for auto network and 2 hidden layers for predictor network)
hidden_size1 = 10
hidden_size2 = 10
assert hidden_size1<=input_size
linear_layer1_for_m3 = Linear(input_size, hidden_size1)
sigmoid_layer1_for_m3 = Linear(hidden_size1, hidden_size1)
linear_layer2_for_m3 = Linear(hidden_size1, hidden_size2)
sigmoid_layer2_for_m3 = Linear(hidden_size2, hidden_size2)
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
