import tensorflow as tf
import numpy as np
from tensorflow import keras

#A simple 5 layer neural network written using keras. 
#It takes a simple 1d array as input and outputs a 1d array of the same size.
#===============================================================================
class NeuralNet(keras.Model):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__(name='NeuralNet')
        self.layers = [
            keras.layers.InputLayer(input_shape=(input_size,)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(output_size, activation='linear'),
            keras.layers.Dense(output_size, activation='linear')
        ]
#        self.model = keras.Sequential(layers)
    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
#        return self.model(inputs, **kwargs)
    #---------------------------------------------------------------------------
#===============================================================================

def kerasmodel():
    input_size = 1
    output_size = 1
    layers = [
                keras.layers.Dense(3, activation='relu', input_shape=(input_size,)),
                keras.layers.Dense(15, activation='relu'),
                keras.layers.Dense(15, activation='sigmoid'),
                keras.layers.Dense(output_size, activation='linear'),
            ]
    model = keras.Sequential(layers)
    model.compile()
    return model
