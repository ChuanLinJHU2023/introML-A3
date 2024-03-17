from DataPreprocessing import *
from DataPredictors import *
from DataLayers import *
import numpy as np
hidden_layer1 = Linear(input_size, hidden_size1)
hidden_layer1_ = Sigmoid(hidden_size1, hidden_size1)

layers = [hidden_layer1, hidden_layer2, output_layer]
seq = Sequential(layers)