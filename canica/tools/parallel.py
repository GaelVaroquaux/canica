"""
Helpers for embarassingly parallel code.
"""

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

import itertools
import pickle

class PMap(object):
    """ Functor to apply map in parallel or not transparently.
    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs    

    def __call__(self, function, *arg_list, **kwargs):
        # Try to pickle the input function, to catch the problems early,
        # in the main thread, rather than in the child processes, where
        # error management is impossible.
        pickle.dumps(function)

        n_jobs = self.n_jobs
        if n_jobs is None or multiprocessing is None or n_jobs == 1:
            n_jobs = 1
            from __builtin__ import apply
        else:
            pool = multiprocessing.Pool(n_jobs)
            apply = pool.apply_async
        output = list()
        for these_arguments in itertools.izip(*arg_list):
            output.append(apply(function, these_arguments, kwargs))
        if n_jobs > 1:
            output = [job.get() for job in output]
            pool.close()
            pool.join()
        return output
 


