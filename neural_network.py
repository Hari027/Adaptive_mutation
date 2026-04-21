# neural_network.py
import numpy as np
import config

class NeuralNetwork:
    """Fully-connected feed-forward NN with tanh hidden, softmax output."""

    def __init__(self, layer_sizes=config.LAYER_SIZES):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
            else:
                e = np.exp(x - x.max())
                x = e / e.sum()
        return x

    def get_flat(self):
        return np.concatenate([w.flatten() for w in self.weights] +
                               [b.flatten() for b in self.biases])

    def set_flat(self, flat):
        idx = 0
        for i in range(len(self.weights)):
            sz = self.weights[i].size
            self.weights[i] = flat[idx:idx+sz].reshape(self.weights[i].shape)
            idx += sz
        for i in range(len(self.biases)):
            sz = self.biases[i].size
            self.biases[i] = flat[idx:idx+sz]
            idx += sz

    def clone(self):
        nn = NeuralNetwork(self.layer_sizes)
        nn.set_flat(self.get_flat().copy())
        return nn
