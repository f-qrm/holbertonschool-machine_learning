#!/usr/bin/env python3
"""Module that represents a Poisson distribution"""


class Poisson:
    """Represents a Poisson distribution."""
    def __init__(self, data=None, lambtha=1.):
        """Initializes a Poisson distribution instance.

            Args:
                data (list): data to estimate the distribution.
                Defaults to None.
                lambtha (float): expected number of occurrences.
                Defaults to 1.

            Raises:
                TypeError: if data is not a list.
                ValueError: if lambtha is not positive or data has less
                than 2 values.
        """
        self.lambtha = float(lambtha)
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            estim_lambtha = sum(data) / len(data)
            self.lambtha = float(estim_lambtha)

        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        """Calculates the PMF value for a given number of successes.

            Args:
                k (int): number of successes.

            Returns:
                float: PMF value for k, or 0 if k is out of range.
            """
        if not isinstance(k, (int)):
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        factorial = 1
        for i in range(1, k+1):
            factorial = factorial * i
        pmf = ((self.lambtha**k) * (e ** (-self.lambtha))) / factorial
        return pmf
