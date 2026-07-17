from scipy import signal
import numpy as np
import pickle as pkl
from Utils import *

class Network:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.001, batch_size=32, verbose=True):
        n_samples = len(x_train)
        for e in range(epochs):
            indices = np.random.permutation(n_samples)
            error = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x_batch = [x_train[idx] for idx in batch_indices]
                y_batch = [y_train[idx] for idx in batch_indices]

                batch_error = 0
                for x, y in zip(x_batch, y_batch):
                    output = self.predict(x)
                    batch_error += loss(y, output)
                    grad = y
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad, learning_rate, accumulate=True)

                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update()

                error += batch_error

            error /= n_samples
            if verbose:
                print(f"{e + 1}/{epochs}, error={error:.6f}")

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pkl.dump(self, f)
        print(f"Saved to {path}")

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            model = pkl.load(f)
        print(f"Loaded from {path}")
        return model


class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))

        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

        self.acc_w_grad = np.zeros_like(self.weights)
        self.acc_b_grad = np.zeros_like(self.bias)
        self.acc_count = 0

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate=None, accumulate=False):
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)

        if accumulate:
            self.acc_w_grad += weights_gradient
            self.acc_b_grad += bias_gradient
            self.acc_count += 1
        else:
            self.t += 1

            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_gradient ** 2)
            m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
            self.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradient
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (bias_gradient ** 2)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
            self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return input_gradient

    def update(self):
        if self.acc_count == 0:
            return

        avg_w_grad = self.acc_w_grad / self.acc_count
        avg_b_grad = self.acc_b_grad / self.acc_count

        self.t += 1

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * avg_w_grad
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (avg_w_grad ** 2)
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        self.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * avg_b_grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (avg_b_grad ** 2)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        self.acc_w_grad.fill(0)
        self.acc_b_grad.fill(0)
        self.acc_count = 0


class Convolutional:
    def __init__(self, input_shape, kernel_size, depth, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        in_depth, h, w = input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.in_depth = in_depth
        self.output_shape = (depth, h - kernel_size + 1, w - kernel_size + 1)

        self.kernels = np.random.randn(depth, in_depth, kernel_size, kernel_size) \
                       * np.sqrt(2.0 / (in_depth * kernel_size * kernel_size))
        self.biases = np.zeros(self.output_shape)

        self.m_k = np.zeros_like(self.kernels)
        self.v_k = np.zeros_like(self.kernels)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

        self.acc_k_grad = np.zeros_like(self.kernels)
        self.acc_b_grad = np.zeros_like(self.biases)
        self.acc_count = 0

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for f in range(self.depth):
            for d in range(self.in_depth):
                self.output[f] += signal.correlate2d(self.input[d], self.kernels[f, d], mode="valid")
        
        return self.output

    def backward(self, output_gradient, learning_rate=None, accumulate=False):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)

        for f in range(self.depth):
            for d in range(self.in_depth):
                kernels_gradient[f, d] = signal.correlate2d(self.input[d], output_gradient[f], mode="valid")
                input_gradient[d] += signal.convolve2d(output_gradient[f], self.kernels[f, d], mode="full")

        bias_gradient = output_gradient

        if accumulate:
            self.acc_k_grad += kernels_gradient
            self.acc_b_grad += bias_gradient
            self.acc_count += 1
        else:
            self.t += 1
            lr = learning_rate if learning_rate is not None else self.lr

            self.m_k = self.beta1 * self.m_k + (1 - self.beta1) * kernels_gradient
            self.v_k = self.beta2 * self.v_k + (1 - self.beta2) * (kernels_gradient ** 2)
            m_k_hat = self.m_k / (1 - self.beta1 ** self.t)
            v_k_hat = self.v_k / (1 - self.beta2 ** self.t)

            self.kernels -= lr * m_k_hat / (np.sqrt(v_k_hat) + self.epsilon)

            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradient
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (bias_gradient ** 2)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            self.biases -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return input_gradient

    def update(self):
        if self.acc_count == 0:
            return

        avg_k_grad = self.acc_k_grad / self.acc_count
        avg_b_grad = self.acc_b_grad / self.acc_count

        self.t += 1

        self.m_k = self.beta1 * self.m_k + (1 - self.beta1) * avg_k_grad
        self.v_k = self.beta2 * self.v_k + (1 - self.beta2) * (avg_k_grad ** 2)
        m_k_hat = self.m_k / (1 - self.beta1 ** self.t)
        v_k_hat = self.v_k / (1 - self.beta2 ** self.t)

        self.kernels -= self.lr * m_k_hat / (np.sqrt(v_k_hat) + self.epsilon)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * avg_b_grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (avg_b_grad ** 2)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        self.biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        self.acc_k_grad.fill(0)
        self.acc_b_grad.fill(0)
        self.acc_count = 0


class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate=None, accumulate=False):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate=None, accumulate=False):
        return np.reshape(output_gradient, self.input_shape)


class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)


class Softmax:
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, y_true, learning_rate=None, accumulate=False):
        return self.output - y_true