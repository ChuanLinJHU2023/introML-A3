import pandas as pd
from DataLossers import *
from DataPredictors import *
from DataOptimizers import *

example_xs= \
    [[-3, 1, -10],[-7, -1, 2],[8, -2, -4],[-4, 7, -9],[-9, 4, 0],
     [-6, 6, -6],[4, 0, -6],[-2, 4, 0],[-1, -4, 7], [0, -8, 6],
     [-6, 3, -10],[-8, 10, -5],[6, -2, -5],[-10, 3, -9],[-7, 2, -1],
     [1, -2, -4],[-10, -4, 6],[-10, -5, 7],[-5, -9, 4],[1, -4, 0]]
def func(a,b,c):
    if a>0:
        return 0
    elif b>0:
        return 1
    elif c>0:
        return 2

example_ys=[func(a,b,c) for (a,b,c) in example_xs]
example_df=pd.DataFrame([(func(a,b,c), a, b, c) for (a,b,c) in example_xs])
print("example xs")
print(example_xs)
print("example ys")
print(example_ys)
print("example df")
print(example_df)
print("\n\n")

input_size = 3
output_size = 3
lr = 0.01
n_epochs = 2000
label_feature = 0
batch_size=10
loss=SoftmaxCrossEntropy(output_size=output_size)
opt=GradientDecsent(lr=lr)
linear_layer = Linear(input_size, output_size)
layers = [linear_layer]
seq = Sequential(layers)
model = DNN(seq, opt, loss, label_feature=label_feature)
model.fit(example_df, batch_size=batch_size,n_epochs=n_epochs)
print(linear_layer.param("W"))
print(linear_layer.param("b"))
model.predict(example_df, show_detail=True)
print("answers")
print(example_ys)
