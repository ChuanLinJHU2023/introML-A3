from DataOptimizers import *
from DataLayers import *
from DataLossers import *
import numpy as  np
import random

data_size=20
def tuple_generate():
    return [random.randint(0,10),random.randint(0,10),random.randint(0,10)]
example_xs=[ tuple_generate() for i in range(data_size)]
example_ys=[a+2*b+3*c for (a,b,c) in example_xs]


input_size=3
output_size=1
lr=0.001
n_epochs=20
W=np.ones((output_size,input_size))
b=np.zeros((output_size,1))
linear_layer=Linear(input_size,output_size,W=W,b=b)
opt=GradientDecsent(lr)
loss=SSE(output_size)
for epoch_i in range(n_epochs):
    for dp_i in range(data_size):
        x=example_xs[dp_i]
        y=example_ys[dp_i]
        y_pred=linear_layer.forward(x,need_reshape=True)
        value=loss.value(y_pred,y)
        grad=loss.grad()
        linear_layer.backward(grad)
    opt.step(linear_layer)
    print(linear_layer.param("W"))



