import numpy as np
import pickle
from scipy import signal
from utils import * 

class Layer:
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError


class Network:
    def __init__(self, layers: list):
        self.layers = layers

    def predict(self, input):
        output = input

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward pass
                output = self.predict(x)
                error += loss(y, output)

                if isinstance(self.layers[-1], SoftmaxCCE):
                    grad = self.layers[-1].backward(y)
                    for layer in reversed(self.layers[:-1]):
                        grad = layer.backward(grad, learning_rate)
                else:
                    grad = loss_prime(y, output)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error:.6f}")

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved to {path}")

    @staticmethod
    def load(path="model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)

        print(f"Loaded from {path}")
        return model


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation       = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class ReLU(Activation):
    def __init__(self):
        super().__init__(_relu, _relu_prime)


class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        
        return self.output

    def backward(self, output_gradient):
        n = self.output.size

        return np.dot(
            (np.identity(n) - self.output.T) * self.output,
            output_gradient
        )


class SoftmaxCCE(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)

        return self.output

    def backward(self, y_true):
        return self.output - y_true


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) \
                       * np.sqrt(2.0 / input_size)
        self.bias    = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient   = np.dot(self.weights.T, output_gradient)

        self.weights    -= learning_rate * weights_gradient
        self.bias       -= learning_rate * output_gradient

        return input_gradient


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        in_depth, h, w    = input_shape
        self.input_shape  = input_shape
        self.kernel_size  = kernel_size
        self.depth        = depth
        self.in_depth     = in_depth
        self.output_shape = (depth, h - kernel_size + 1, w - kernel_size + 1)
        self.kernels = np.random.randn(depth, in_depth, kernel_size, kernel_size) \
                       * np.sqrt(2.0 / (in_depth * kernel_size * kernel_size))
        self.biases  = np.zeros(self.output_shape)

    def forward(self, input):
        self.input  = input
        self.output = np.copy(self.biases)
        for f in range(self.depth):
            for d in range(self.in_depth):
                self.output[f] += signal.correlate2d(
                    self.input[d], self.kernels[f, d], mode="valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient   = np.zeros_like(self.input)
        for f in range(self.depth):
            for d in range(self.in_depth):
                kernels_gradient[f, d] = signal.correlate2d(
                    self.input[d], output_gradient[f], mode="valid"
                )
                input_gradient[d] += signal.convolve2d(
                    output_gradient[f], self.kernels[f, d], mode="full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * output_gradient

        return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape  = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)
