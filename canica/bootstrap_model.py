# XXX !!! This is a modified copy of 'half_split'

import os
import random

import numpy as np

from validate_model import generate_icas, this_config

from joblib import PrintTime
from rs.io_context import IOContext

## Avoid side effects with other bits of code.
#this_config = this_config.copy()
#this_config['output_dir'] = '~/data-nonsync/half_split'
#this_config = IOContext(extra_options=this_config)

print_time = PrintTime()

def do_ica(subjects):
    _, mask, fmri_header, ica = generate_icas(subjects)
    icas = list()
    for this_ica in ica:
        norm = np.sqrt((this_ica**2).sum())
        if norm > 0:
            this_ica = this_ica/norm
            icas.append(this_ica)
    return np.array(icas)


def do_ica_memoized(subjects):
    subjects.sort()
    hash = '-'.join(['%02i' % s for s in subjects])
    filename = this_config.outpath('icas_%s.npy' % hash)
    print 'Bootstrapping results in %s' % filename
    if not os.path.exists(filename):
        icas = do_ica(subjects)
        np.save(filename, icas)
    return filename


def bootstrap_func(n_samples, func, n_jobs=1):
    """ Call 'func' with subjects bootstrapped.
    """
    if n_jobs > 1:
        from multiprocessing import Pool
        pool = Pool(7)
        my_apply = pool.apply_async
    else:
        import __builtin__
        my_apply = __builtin__.apply

    jobs = list()
    for i in range(n_samples):
        subjects = [random.choice(this_config.subjects)
                        for _ in range(11)]
        jobs.append(my_apply(func, (subjects, )),)
        print 'Ran %s on subjects %s' % (func, subjects)

    if n_jobs>1:
        for job in jobs:
            job.get()
        pool.close()
        pool.join()

if __name__ == '__main__':
    bootstrap_func(30, do_ica_memoized, n_jobs=7)

