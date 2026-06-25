#!/usr/bin/env python3
"""Module that represents an Exponential distribution."""


class Exponential:
    """Represents an Exponential distribution."""
    def __init__(self, data=None, lambtha=1.):
        """Initializes an Exponential distribution instance.

        Args:
            data (list): data to estimate the distribution. Defaults to None.
            lambtha (float): expected rate of occurrences. Defaults to 1.

        Raises:
            TypeError: if data is not a list.
            ValueError: if lambtha is not positive or data has less than
            2 values.
        """
        self.lambtha = float(lambtha)
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            estim_lambtha = 1 / (sum(data) / len(data))
            self.lambtha = float(estim_lambtha)
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
