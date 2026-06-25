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

    def pdf(self, x):
        """Calculates the PDF value for a given time period.

            Args:
                x (float): time period.

            Returns:
                float: PDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        pdf = self.lambtha * e**(-self.lambtha * x)
        return pdf

    def cdf(self, x):
        """Calculates the CDF value for a given time period.

            Args:
                x (float): time period.

            Returns:
                float: CDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        cdf = 1 - (e ** (-self.lambtha * x))
        return cdf
