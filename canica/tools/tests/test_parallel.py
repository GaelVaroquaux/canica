""" 
Tests for the parallel helpers. 
"""
import pickle

import nose

from ..parallel import Parallel, delayed

def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x**2 + y + z

def test_pmap():
    """ Check that pmap works OK, with and without multiprocessing.
    """
    lst = range(10)
    for n_jobs in (1, 4):
        yield (nose.tools.assert_equal, 
               [f(x) for x in lst], 
               Parallel(n_jobs=n_jobs)(delayed(f)(x) for x in lst)
               )


def test_pmap_kwargs():
    """ Check the keyword argument processing of pmap.
    """
    lst = range(10)
    for n_jobs in (1, 4):
        yield (nose.tools.assert_equal, 
               [f(x, y=1) for x in lst], 
               Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst)
              )

        
def test_pmap_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    def g(x):
        return x**2
    nose.tools.assert_raises(pickle.PicklingError,
                             Parallel(), 
                             (delayed(g)(x) for x in range(10))
                            )


