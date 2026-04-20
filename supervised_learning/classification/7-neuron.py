#!/usr/bin/env python3
""" Class that defines a single neuron performing classification """
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ Class that defines a single neuron performing classification """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Return the weight """
        return self.__W

    @property
    def b(self):
        """ Return the bias """
        return self.__b

    @property
    def A(self):
        """ Return the prediction """
        return self.__A

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculate the cost of the neuron using logistic regression """
        m = Y.shape[1]
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        mu = np.sum(L) / m
        return mu

    def evaluate(self, X, Y):
        """ Calculate the neuron`s prediction and the cost """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Update W et b to reduce the loss"""
        m = X.shape[1]
        dZ = A - Y
        dW = (1/m) * np.matmul(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        new_W = self.__W - alpha * dW
        new_b = self.__b - alpha * db
        self.__W = new_W
        self.__b = new_b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Train the neuron over a number of iterations, updating W, b and A.
            Prints cost every step iterations if verbose is True.
            Plots training cost over iterations if graph is True.
            Returns the evaluation of the training data after training. """
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
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        if verbose:
            print(f"Cost after 0 iterations: {cost}")
        iters = [0]
        loss = [cost]
        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
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
