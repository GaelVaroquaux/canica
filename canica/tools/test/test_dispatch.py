""" 
Tests for the dispatch iterator.
"""
import random
import nose

from ..dispatch import dispatch

def test_dispatch():
    for _ in range(10):
        n = random.randint(50, 100)
        n_packs = random.randint(1, 10)
        n_ = 0
        for this_n in dispatch(n, n_packs):
            n_ += this_n
        yield nose.tools.assert_equal, n_, n
        

