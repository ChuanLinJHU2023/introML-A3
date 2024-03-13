import numpy as np
from DataPredictors import *


# input_size=4
# output_size=2
# linear_layer=Linear(input_size,output_size,examplary=True)
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


input_size=4
inter_size=2
output_size=3
linear_layer1=Linear(input_size,inter_size,examplary=True)
linear_layer2=Linear(inter_size,output_size,examplary=True)
layers=[linear_layer1,linear_layer2]
seq=Sequential(layers)
input=np.arange(input_size).reshape((input_size,1))
print(seq.forward(input))
print("\n\n")
output_gradient=np.arange(output_size).reshape((1,output_size))
seq.backward(output_gradient)
print(seq.layers[0].grad("W"))
print(seq.layers[1].grad("W"))