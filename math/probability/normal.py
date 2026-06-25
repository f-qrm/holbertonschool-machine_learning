#!/usr/bin/env python3
"""Module that represents a Normal distribution."""


class Normal:
    """Represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes a Normal distribution instance.

        Args:
            data (list): data to estimate the distribution. Defaults to None.
            mean (float): mean of the distribution. Defaults to 0.
            stddev (float): standard deviation of the distribution.
            Defaults to 1.

        Raises:
            TypeError: if data is not a list.
            ValueError: if stddev is not positive or data has less
            than 2 values.
        """
        self.stddev = float(stddev)
        self.mean = float(mean)
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = 0
            for xi in data:
                variance += (xi - self.mean) ** 2
            variance = variance / len(data)
            self.stddev = (variance ** 0.5)
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

    def z_score(self, x):
        """Calculates the z-score of a given x-value.

            Args:
                x (float): x-value.

            Returns:
                float: z-score of x.
        """
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """Calculates the x-value of a given z-score.

            Args:
                z (float): z-score.

            Returns:
                float: x-value of z.
        """
        x = self.mean + z * self.stddev
        return x

    def pdf(self, x):
        """Calculates the PDF value for a given x-value.

            Args:
                x (float): x-value.

            Returns:
                float: PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536
        pdf = (1 / (self.stddev * (2 * pi) ** 0.5)) * (
            e ** (-0.5 * self.z_score(x) ** 2))
        return pdf

    def cdf(self, x):
        """Calculates the CDF value for a given x-value.

            Args:
                x (float): x-value.

            Returns:
                float: CDF value for x.
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / (pi ** 0.5)) * (z - ((z**3) / 3) + ((z**5) / 10)
                                   - ((z**7) / 42) + ((z**9) / 216))
        cdf = 0.5 * (1 + erf)
        return cdf
