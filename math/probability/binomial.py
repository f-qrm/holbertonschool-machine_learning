#!/usr/bin/env python3
"""Module that represents a Binomial distribution."""


class Binomial:
    """Represents a Binomial distribution."""
    def __init__(self, data=None, n=1, p=0.5):
        """Initializes a Binomial distribution instance.

        Args:
            data (list): data to estimate the distribution. Defaults to None.
            n (int): number of Bernoulli trials. Defaults to 1.
            p (float): probability of a success. Defaults to 0.5.

        Raises:
            TypeError: if data is not a list.
            ValueError: if n is not positive, p is not valid,
            or data has less than 2 values.
        """
        self.n = int(n)
        self.p = float(p)
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = 0
            for xi in data:
                variance += (xi - mean) ** 2
            variance = variance / len(data)
            p = 1 - (variance / mean)
            self.n = round(mean / p)
            self.p = mean / self.n

        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
