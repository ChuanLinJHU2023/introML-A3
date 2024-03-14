from DataOptimizers import *
from DataLayers import *
from DataLossers import *
import numpy as  np
import random

data_size=20
def tuple_generate():
    return [random.randint(0,10),random.randint(0,10),random.randint(0,10)]
example_xs=[ tuple_generate() for i in range(data_size)]
example_xs=\
    [[3, 1, 10], [7, 1, 2], [8, 2, 4], [4, 7, 9], [9, 4, 0], [6, 6, 6],
     [4, 0, 6], [2, 4, 0], [1, 4, 7], [0, 8, 6], [6, 3, 10], [8, 10, 5],
     [6, 2, 5], [10, 3, 9], [7, 2, 1], [1, 2, 4], [10, 4, 6], [10, 5, 7],
     [5, 9, 4], [1, 4, 0]]
example_ys=[a+2*b+3*c for (a,b,c) in example_xs]
print("example xs")
print(example_xs)
print("example ys")
print(example_ys)
print("\n\n")


def expeiment1():
    input_size=3
    output_size=1
    lr=0.00001
    # n_epochs=20
    n_epochs=20000
    W=np.ones((output_size,input_size))
    # W=np.array([1,2,3]).reshape((output_size,input_size)).astype(np.float)
    b=np.zeros((output_size,1))
    linear_layer=Linear(input_size,output_size,W=W,b=b)
    opt=GradientDecsent(lr)
    loss=SSE(output_size)
    for epoch_i in range(n_epochs):
        print("Epoch {}".format(epoch_i)+"//"*100)
        print("W", linear_layer.params[0])
        print("W_grad", linear_layer.grads[0])
        print("\n")
        for dp_i in range(data_size):
            x=example_xs[dp_i]
            y=example_ys[dp_i]
            y_pred=linear_layer.forward(x,need_reshape=True)
            value=loss.value(y_pred,y)
            grad=loss.grad()
            linear_layer.backward(grad)
            # print("x",x)
            # print("y",y)
            # print("y pred",y_pred)
            # print("value", value)
            # print("W",linear_layer.params[0])
            # print("W_grad",linear_layer.grads[0])
            # print("\n")
        opt.step(linear_layer)
# expeiment1()


def expeiment2():
    input_size=3
    output_size=1
    lr=0.00001
    # n_epochs=20
    n_epochs=20000
    W=np.ones((output_size,input_size))
    # W=np.array([1,2,3]).reshape((output_size,input_size)).astype(np.float)
    b=np.zeros((output_size,1))
    linear_layer=Linear(input_size,output_size,W=W,b=b)
    layers=[linear_layer]
    seq=Sequential(layers)
    opt=GradientDecsent(lr)
    loss=SSE(output_size)
    for epoch_i in range(n_epochs):
        print("Epoch {}".format(epoch_i)+"//"*100)
        print("W", linear_layer.params[0])
        print("W_grad", linear_layer.grads[0])
        print("\n")
        for dp_i in range(data_size):
            x=example_xs[dp_i]
            y=example_ys[dp_i]
            y_pred=seq.forward(x,need_reshape=True)
            value=loss.value(y_pred,y)
            grad=loss.grad()
            seq.backward(grad)
            # print("x",x)
            # print("y",y)
            # print("y pred",y_pred)
            # print("value", value)
            # print("W",linear_layer.params[0])
            # print("W_grad",linear_layer.grads[0])
            # print("\n")
        opt.step(linear_layer)
# expeiment2()