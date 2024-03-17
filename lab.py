from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *
import numpy as np
input_size=4
hidden_size1=4
hidden_layer1 = Linear(input_size, hidden_size1)
hidden_layer2 = Linear(input_size, hidden_size1)
hidden_layer1_ = Sigmoid(hidden_size1, hidden_size1)
layers = [hidden_layer1, hidden_layer1_]
seq = Sequential(layers)
layers = [hidden_layer2, hidden_layer1_]
seq2 = Sequential(layers)
print(seq.layers[0] is seq2.layers[0])
print(seq.layers[1] is seq2.layers[1])