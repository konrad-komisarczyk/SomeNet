import numpy as np

class MyNet:

    def __init__(self, layers, biases, activation_f):
        if len(layers) != len(biases):
            raise ValueError
        for i in range(0, len(layers) - 1):
            if layers[i].shape[1] != layers[i + 1].shape[0]:
                raise ValueError
        for i in range(0, len(layers)):
            if biases[i].shape[0] != 1:
                raise ValueError
            if biases[i].shape[1] != layers[i].shape[1]:
                raise ValueError()

        self.layers = layers
        self.biases = biases
        self.activation_f = activation_f

    def predict(self, input):
        if input.shape[1] != self.layers[0].shape[0]:
            raise ValueError
        result = input
        for i in range(0, len(self.layers)):
            f = self.activation_f[i]
            result = f(result @ self.layers[i] + self.biases[i])
        return result
