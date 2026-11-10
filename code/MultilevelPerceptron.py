import numpy as np
import csv


def ReLU(layers):
    if type(layers) == list:
        layers = np.array(layers, dtype=np.float64)
    return np.where(layers < 0, 0, layers)


def ReLUDerivative(layers):
    return np.where(layers < 0, 0, 1)


def train(train_x, train_y, test_x, learning_rate=0.00001, iters=500000):
    input_size = train_x.shape[1]
    N = train_x.shape[0]
    batch_size = N
    layer_sizes = [16, 16, 16] + [1]

    # layer setup
    weights = [np.zeros((input_size, layer_sizes[0]), dtype=np.float64)]
    biases = []
    layers = [np.zeros((batch_size, input_size), dtype=np.float64)]
    for x in layer_sizes:
        biases.append(np.ones((x), dtype=np.float64))
        layers.append(np.zeros((batch_size, x), dtype=np.float64))

    for i in range(0, len(layer_sizes) - 1):
        weights.append(np.zeros((layer_sizes[i], layer_sizes[i + 1]), dtype=np.float64))

    # training
    for iter in range(iters):
        indexes = np.random.choice(range(N), size=batch_size, replace=False)
        layers[0] = train_x[indexes]
        expected = train_y[indexes].reshape(-1, 1)

        # forward prop
        for i in range(len(layers) - 1):
            if i < len(layers) - 1:
                layers[i + 1] = ReLU(layers[i].dot(weights[i]) + biases[i])
            else:
                layers[i + 1] = layers[i].dot(weights[i]) + biases[i]

        # (the print of the error on every iteration is left out)

        # backward prop
        bias_diff = [np.zeros_like(x) for x in biases]
        weights_diff = [np.zeros_like(x) for x in weights]

        delta = 2 * (layers[-1] - expected)
        for i in range(len(layer_sizes) - 1, -1, -1):
            weights_diff[i] = layers[i].T @ delta / batch_size
            bias_diff[i] = delta.mean(axis=0)

            if i > 0:
                delta = delta @ weights[i].T * ReLUDerivative(layers[i])

        for x in range(len(layer_sizes) - 1, -1, -1):
            weights[x] = weights[x] - learning_rate * weights_diff[x]
            biases[x] = biases[x] - learning_rate * bias_diff[x]

    return weights, biases


def load_weights():
    w = np.load("MLP_weights.npz")
    b = np.load("MLP_biases.npz")
    weights = []
    biases = []

    for x in range(len(w)):
        weights.append(w[f"arr_{x}"])
        biases.append(b[f"arr_{x}"])

    return weights, biases


def run(train_x, train_y, test_x, train=False, layer_sizes=[16, 16, 16, 1], **kwargs):
    if train:
        weights, biases = train()
    else:
        weights, biases = load_weights()

    layers = [test_x]
    for x in layer_sizes:
        layers.append(np.zeros((test_x.shape[0], x), dtype=np.float64))

    # forward prop
    for i in range(len(layers) - 1):
        if i < len(layers) - 1:
            layers[i + 1] = ReLU(layers[i].dot(weights[i]) + biases[i])
        else:
            layers[i + 1] = layers[i].dot(weights[i]) + biases[i]

    return layers[-1].reshape(1, -1)
