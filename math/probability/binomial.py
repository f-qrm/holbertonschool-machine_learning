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

    def pmf(self, k):
        """Calculates the PMF value for a given number of successes.

            Args:
                k (int): number of successes.

            Returns:
                float: PMF value for k, or 0 if k is out of range.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        factorial_k = 1
        for i in range(1, k+1):
            factorial_k = factorial_k * i
        factorial_n = 1
        for i in range(1, self.n+1):
            factorial_n = factorial_n * i
        n_k = self.n - k
        factorial_nk = 1
        for i in range(1, n_k+1):
            factorial_nk = factorial_nk * i
        coef = factorial_n / (factorial_k * factorial_nk)
        pmf = coef * self.p ** k * (1 - self.p) ** n_k
        return pmf

    def cdf(self, k):
        """Calculates the CDF value for a given number of successes.

            Args:
                k (int): number of successes.

            Returns:
                float: CDF value for k, or 0 if k is out of range.
        """
        if not isinstance(k, (int)):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k+1):
            cdf += self.pmf(i)
        return cdf
