""" 
Tests for the dispatch iterator.
"""
import random
import nose

from ..dispatch import dispatch, dispatch_lst

def test_dispatch():
    for _ in range(10):
        n = random.randint(50, 100)
        n_packs = random.randint(1, 10)
        n_ = 0
        for this_n in dispatch(n, n_packs):
            n_ += this_n
        yield nose.tools.assert_equal, n_, n
        

def test_dispatch_lst():
    orig_set = set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    n_packs = 3
    gathered_set = set()
    for this_lst in dispatch_lst(orig_set, n_packs):
        gathered_set.update(this_lst)
    nose.tools.assert_equal, gathered_set, orig_set
 
