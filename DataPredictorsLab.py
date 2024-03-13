import numpy as np
from DataPredictors import *


# input_size=4
# output_size=2
# W=np.arange(output_size*input_size).reshape((output_size,input_size))
# b=np.zeros((output_size,1))
# linear_layer=Linear(input_size,output_size,W=W,b=b)
# input=np.arange(input_size).reshape((input_size,1))
# print(input)
# print(linear_layer.param("W"))
# print(linear_layer.forward(input))
# print("\n\n")
# output_gradient=np.arange(output_size).reshape((1,output_size))
# print(output_gradient)
# print(linear_layer.backward(output_gradient))
# print(linear_layer.grad("W"))
# print(linear_layer.grad("b"))

# input_size=4
# output_size=2
# linear_layer=Linear(input_size,output_size)
# input=np.ones((input_size,1))
# print(linear_layer.forward(input))
# print(linear_layer.param("W"))
# print(linear_layer.param("b"))