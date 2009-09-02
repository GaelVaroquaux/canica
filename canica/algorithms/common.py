"""
Common tools for statistical analysis.
"""

# Author: Gael Varoquaux
# License: BSD 3 clause
import numpy as np


def center_and_norm(x, axis=-1):
    """ Centers and norms x **in place**

        Parameters
        -----------
        x: ndarray
            Array with an axis of observations (statistical units) measured on 
            random variables.
        axis: int, optionnal
            Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)

