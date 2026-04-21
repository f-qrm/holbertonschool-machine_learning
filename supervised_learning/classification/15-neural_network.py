#!/usr/bin/env python3
""" Neural Network """
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ defines a neural network with one hidden layer
        performing binary classification """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Return the weight of the hidden layer """
        return self.__W1

    @property
    def b1(self):
        """ Return the biais of the hidden layer """
        return self.__b1

    @property
    def A1(self):
        """ Return the activated output of the hidden layer """
        return self.__A1

    @property
    def W2(self):
        """ Return the weight of the output neuron """
        return self.__W2

    @property
    def b2(self):
        """ Return the biai of the output neuron """
        return self.__b2

    @property
    def A2(self):
        """ Return the activated output of the output neuron """
        return self.__A2

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron and
            the output neuron """
        Z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z))
        Z = np.matmul(self.__W2, self.A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculate the cost of the neuron using logistic regression """
        m = Y.shape[1]
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        mu = np.sum(L) / m
        return mu

    def evaluate(self, X, Y):
        """ Calculate the neural network's prediction and the cost" """
        __, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculate gradient descent for the output layer then the
            hidden layer """
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        new_W2 = self.__W2 - alpha * dW2
        new_b2 = self.__b2 - alpha * db2
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1/m) * np.matmul(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        new_W1 = self.__W1 - alpha * dW1
        new_b1 = self.__b1 - alpha * db1
        self.__W2 = new_W2
        self.__b2 = new_b2
        self.__W1 = new_W1
        self.__b1 = new_b1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Main method that orchestrates the training by repeating forward
            propagation and gradient descent over a number of iterations. """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        if verbose:
            print(f"Cost after 0 iterations: {cost}")
        iters = [0]
        loss = [cost]
        for i in range(1, iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cost = self.cost(Y, A2)
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
