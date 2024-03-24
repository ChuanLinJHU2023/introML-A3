import pandas as pd

from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *

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
print("The processed dataset is:")
print(df, "\n\n")


label_feature = 6
type = "c"
input_size = df.shape[1] - 1
output_size = 4
print("The input size is {}".format(input_size))
print("The output size is {}\n\n".format(output_size))


lr = 0.01
n_epochs = 50
batch_size = 50
loss = SoftmaxCrossEntropy(output_size=output_size)
opt = GradientDecsent(lr=lr)

hidden_size=10
hidden_layer = Linear(input_size, hidden_size)
hidden_layer_ = Sigmoid(hidden_size, hidden_size)
output_layer = Linear(hidden_size, output_size)
layers = [hidden_layer, hidden_layer_, output_layer]
seq = Sequential(layers)

output_layer_decode = Linear(hidden_size, input_size)
layers_enc = [hidden_layer, hidden_layer_, output_layer_decode]
seq_enc = Sequential(layers_enc)

model = AutoEncoder(seq_enc, seq, opt, loss,
                    label_feature=label_feature, batch_size=batch_size, n_epochs=n_epochs, clear_after_pred=False)

model.fit(df)
prediction=model.predict(df)[:50]
answer=df[label_feature][:50]
print(prediction)
print(list(answer))
data_prediction_evaluator(answer,prediction,metric="01Loss",show_detail=True)


model.fit(df)
prediction=model.predict(df)[:50]
answer=df[label_feature][:50]
print(prediction)
print(list(answer))
data_prediction_evaluator(answer,prediction,metric="01Loss",show_detail=True)


