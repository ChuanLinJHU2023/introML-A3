import numpy as np
import pandas as pd

from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *
from DataValidators import *

df=data_loader("Datasets/breast-cancer-wisconsin.data",missing_value_symbol="?",idCol=0)
print("House Dataset")
df=data_loader("Datasets/house-votes-84.data")
print("The original dataset is:")
print(df,"\n\n")
for i in range(1,17):
    df=data_encode(df,i,"o",["n","?","y"])
df = data_label_rename(df,0)
df=data_standardizer(df,list(range(1,17)))
print("The processed dataset is:")
print(df,"\n\n")
label_feature = 0
model_type = "c"
metric = "MSE" if model_type == "r" else "01Loss"
print("the class labels are:", np.unique(df[10]))
print("the shape of df is ",df.shape)
# print("the shape of df is ",df.iloc[:,1:].shape)


# For the following models
input_size = df.shape[1] - 1
output_size = 4
lr = 0.001
n_epochs = 1
batch_size = 50
print("The input size is {}".format(input_size))
print("The output size is {}".format(output_size))
print("The learning rate is {}".format(lr))
print("The epoch number is {}".format(n_epochs))
print("The batch size is {}\n\n".format(batch_size))
loss = SoftmaxCrossEntropy(output_size=output_size) if model_type == "c" else SSE(output_size=output_size)
opt = GradientDecsent(lr=lr)


# The first model is the Null model
null_model = Null(model_type, label_feature=label_feature)


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
FNN_model = DNN(seq_for_m2, opt, loss, label_feature=label_feature, batch_size=batch_size,
                n_epochs=n_epochs, clear_after_pred=True, show_detials=False)
print(len(FNN_model.seq.layers))

# Autoencoder Model (1 hidden layer for auto network and 2 hidden layers for predictor network)
hidden_size1 = 10
hidden_size2 = 20
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

lr=0.001
opt = GradientDecsent(lr=lr)
AE_model = AutoEncoder(seq_for_m3_enc, seq_for_m3, opt, loss,
                    label_feature=label_feature, batch_size=batch_size, n_epochs=n_epochs, clear_after_pred=True, show_details=False)




# Here is the experiment
predictors = [null_model, linear_model, FNN_model, AE_model]
ratio_list=[50,50]
df80, df20=data_partitioner(df,ratioList=ratio_list)

print("\n\n\n\n")
dp = df20.iloc[0]
print("the data point is:")
print(dp)

print("\n\n\n\n")
print("The is LM")
linear_model.opt.show_weight_update_process = True
linear_model.fit(df80)

print("\n\n\n\n"*3)
print("The is FNN")
FNN_model.opt.show_weight_update_process = True
FNN_model.fit(df80)

print("\n\n\n\n"*3)
print("The is AE")
AE_model.opt.show_weight_update_process = True
AE_model.fit(df80)








