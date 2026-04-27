#!/usr/bin/env python3
"""Module containing the DeepNeuralNetwork class"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l_number in range(1, self.L + 1):
            if not isinstance(layers[l_number - 1], int) or\
                              layers[l_number - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            if l_number == 1:
                prev_nodes = nx
            else:
                prev_nodes = layers[l_number - 2]
            self.weights['W' + str(l_number)] = (
                np.random.randn(layers[l_number - 1], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )
            self.weights['b' + str(l_number)] = (
                np.zeros((layers[l_number - 1], 1))
            )

    @property
    def activation(self):
        """ Getter for the activation function type. """
        return self.__activation

    @property
    def L(self):
        """Getter for the number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Args:
                X (numpy.ndarray): input data with shape (nx, m).

            Returns:
                tuple: the output of the neural network and the cache.
        """
        self.__cache['A0'] = X
        for l_number in range(1, self.L + 1):
            a_prev = self.__cache['A' + str(l_number - 1)]
            W = self.weights['W' + str(l_number)]
            b = self.weights['b' + str(l_number)]
            Z = np.dot(W, a_prev) + b
            if l_number == self.L:
                e_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                a_sig = e_z / np.sum(e_z, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    a_sig = 1 / (1 + np.exp(-Z))
                else:
                    a_sig = np.tanh(Z)
            self.__cache['A' + str(l_number)] = a_sig
        return a_sig, self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y (numpy.ndarray): correct labels with shape (1, m).
                A (numpy.ndarray): activated output of the neuron with
                shape(1, m).

            Returns:
                float: the cost of the model.
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.

            Args:
                X (numpy.ndarray): input data with shape (nx, m).
                Y (numpy.ndarray): correct labels with shape (1, m).

            Returns:
                tuple: the predicted labels and the cost of the network.
        """
        pred, _ = self.forward_prop(X)
        prediction = np.eye(pred.shape[0])[np.argmax(pred, axis=0)].T
        cost = self.cost(Y, pred)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network.

            Args:
                Y (numpy.ndarray): correct labels with shape (1, m).
                cache (dict): intermediary values of the network.
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.L)] - Y
        for l_number in range(self.__L, 0, -1):
            dW = 1 / m * np.dot(dZ, cache['A' + str(l_number - 1)].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if l_number > 1:
                if self.__activation == 'sig':
                    dZ = (
                        np.dot(self.weights['W' + str(l_number)].T, dZ) *
                        cache['A' + str(l_number - 1)] *
                        (1 - cache['A' + str(l_number - 1)])
                        )
                else:
                    dZ = (
                        np.dot(self.weights['W' + str(l_number)].T, dZ) *
                        (1 - cache['A' + str(l_number - 1)] ** 2)
                    )
            new_W = self.weights['W' + str(l_number)] - alpha * dW
            new_b = self.weights['b' + str(l_number)] - alpha * db
            self.weights['W' + str(l_number)] = new_W
            self.weights['b' + str(l_number)] = new_b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
            Trains the deep neural network.

            Args:
                X (numpy.ndarray): input data with shape (nx, m).
                Y (numpy.ndarray): correct labels with shape (1, m).
                iterations (int): number of iterations to train over.
                alpha (float): learning rate.

            Returns:
                tuple: the evaluation of the training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 and step > iterations:
                raise ValueError("step must be positive and <= iterations")
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            iters = [0]
            loss = [cost]
        if verbose:
            print(f"Cost after 0 iterations: {cost}")

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            if verbose and (i % step == 0 or i == iterations):
                iters.append(i)
                loss.append(cost)
                print(f"Cost after {i} iterations: {cost}")
        if graph:
            plt.plot(iters, loss, 'b')
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format.

            Args:
                filename (str): file to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        """
            Loads a pickled DeepNeuralNetwork object.

            Args:
                filename (str): file from which the object should be loaded

            Returns:
                DeepNeuralNetwork: the loaded object, or None if filename
                doesn't exist.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
