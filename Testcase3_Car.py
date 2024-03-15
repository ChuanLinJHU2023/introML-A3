import pandas as pd

from DataPreprocessing import *
from DataPredictors import *

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


hidden_size1=50
hidden_size2=20
lr = 0.001
n_epochs = 500
batch_size = 100
loss = SoftmaxCrossEntropy(output_size=output_size)
opt = GradientDecsent(lr=lr)
hidden_layer1 = Linear(input_size, hidden_size1)
hidden_layer1_ = Sigmoid(hidden_size1, hidden_size1)
hidden_layer2 = Linear(hidden_size1, hidden_size2)
hidden_layer2_ = Sigmoid(hidden_size2, hidden_size2)
output_layer = Linear(hidden_size2, output_size)
layers = [hidden_layer1, hidden_layer2, output_layer]
seq = Sequential(layers)
model = DNN(seq, opt, loss, label_feature=label_feature)
model.fit(df, batch_size=batch_size, n_epochs=n_epochs)
prediction=model.predict(df)[:20]
answer=df[label_feature][:20]
