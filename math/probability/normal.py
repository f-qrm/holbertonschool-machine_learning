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
