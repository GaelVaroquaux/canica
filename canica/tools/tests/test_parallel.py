""" 
Tests for the parallel helpers. 
"""
import pickle

import nose

from ..parallel import PMap

def f(x):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x**2

def test_pmap():
    """ Check that pmap works OK, with and without multiprocessing.
    """
    for n_jobs in (1, 4):
        pmap = PMap(n_jobs=n_jobs)
        yield nose.tools.assert_equal, [x**2 for x in range(10)], \
                                pmap(f, range(10))

        
def test_pmap_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    def g(x):
        return x**2
    nose.tools.assert_raises(pickle.PicklingError,
                             PMap(n_jobs=1), g, range(10))


