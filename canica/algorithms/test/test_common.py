"""
Test the common module.
"""
import nose

import numpy as np

from ..common import center_and_norm

def test_center_and_norm():
    # On a 1D array
    a = np.random.random((100,))
    center_and_norm(a)
    yield np.testing.assert_almost_equal, a.mean(), 0
    yield np.testing.assert_almost_equal, a.std(), 1

    # On a 2D array
    a = np.random.random((100, 2))
    center_and_norm(a)
    yield np.testing.assert_almost_equal, a.mean(axis=-1), 0
    yield np.testing.assert_almost_equal, a.std(axis=-1), 1

