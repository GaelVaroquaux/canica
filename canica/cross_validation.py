"""
Cross validation of ICA group methods.
"""
import copy
import random
import os

# Major scientific libraries import
import numpy as np
from scipy import stats

# Unusual libraries import
from joblib import Memory

# Local imports
from .canica import canica
from .tools.parallel import Parallel, delayed

################################################################################
# Utilities to compare maps 
################################################################################

def find_permutation(X, Y):
    """ Returns the permutation indices of Y that maximise the correlation
        between the vectors of Y and the vectors of X.
    """
    x_list = list()
    y_list = list()
    K = np.dot(X.T, Y)
    permutation = np.zeros((K.shape[0],), int)
    
    for target_index in np.argsort(-np.abs(K).flatten()):
        x, y = np.unravel_index(target_index, K.shape)
        if not x in x_list and not y in y_list:
            x_list.append(x)
            y_list.append(y)
            permutation[x] = y
        if len(x_list) == permutation.shape[0]:
            break
    
    return permutation


################################################################################
# Comparison of ICA runs

def correlation_stats(ica1, ica2):
    """ Calculate statistics on the cross-correlation matrix between 2
        sets of maps.
    """
    permutation = find_permutation(ica1, ica2)
    ica2 = ica2[:, permutation]
    d = float(min(ica1.shape[1], ica2.shape[1]))
    cor = np.abs(np.dot(ica1.T, ica2))
    results = {
            'Frobenius norm': np.trace(np.dot(cor.T, cor))/d,
            'Max correlations': np.hstack((cor.max(axis=0),
                                           cor.max(axis=1))),
            'Trace': np.trace(cor)/d,
        }
    return results


def merge_stats(result_dicts):
    """ Merge a list of results of correlation_stats.
    """
    out_dict = dict()
    for result_dict in result_dicts:
        for key, value in result_dict.iteritems():
            if not key in out_dict:
                out_dict[key] = list()
            out_dict[key].append(value)

    for key, value in out_dict.iteritems():
        out_dict[key] = np.hstack(value)

    return out_dict

def report_stats(stat_dict, working_dir=None):
    """ Save a report of the cross-correlation statistics.
    """
    d = stat_dict
    report = """
    Trace                             : %.2f +/- %.2f
    Frobenius norm                    : %.2f +/- %.2f
    Median of the max component match : %.2f
    Percentile above 0.5              : %i
    Percentile below 0.25             : %i
    Percentile above 0.75             : %i
    """ % (d['Trace'].mean(),
           d['Trace'].std(),
           d['Frobenius norm'].mean(),
           d['Frobenius norm'].std(),
           np.median(d['Max correlations']),
           100 - stats.percentileofscore(d['Max correlations'], .5),
           stats.percentileofscore(d['Max correlations'], .25),
           100 - stats.percentileofscore(d['Max correlations'], .75),
          )
    return report


################################################################################
# Cross validation of canica

def canica_pair(file_pair, n_pca_components, mask, threshold,
                        reference_icas,
                        ccs_threshold=None,
                        n_ica_components=None, do_cca=True,
                        n_jobs=1, working_dir=None):
    ica1, ica2 = [canica(files, n_pca_components, ccs_threshold=ccs_threshold,
                            n_ica_components=n_ica_components, do_cca=do_cca,
                            mask=mask,
                            n_jobs=n_jobs, working_dir=working_dir)
                        for files in file_pair]

    # Separate the maps from the mask and threshold in the canica
    # result.
    ica1, _, _ = ica1
    ica2, _, _ = ica2

    un_thr_stats = correlation_stats(ica1, ica2)

    ica1[np.abs(ica1) < threshold] = 0
    vars = np.sqrt((ica1**2).sum(axis=0))
    vars[vars == 0] = 1
    ica1 /= vars
    ica2[np.abs(ica2) < threshold] = 0
    vars = np.sqrt((ica2**2).sum(axis=0))
    vars[vars == 0] = 1
    ica2 /= vars
    thr_stats = correlation_stats(ica1, ica2)

    reproducibility = 0.5*(  np.dot(reference_icas.T, ica1).max(axis=1)
                           + np.dot(reference_icas.T, ica2).max(axis=1))

    return un_thr_stats, thr_stats, reproducibility
    


def canica_split_half(filenames, n_pca_components, n_split_half=50,
                        ccs_threshold=None, n_ica_components=None, 
                        do_cca=True, mask=None,
                        threshold_p_value=5e-3,
                        n_jobs=1, working_dir=None):
    # First do a full CanICA run, to have references and common mask:
    reference_icas, mask, threshold = canica(filenames, 
                                    n_pca_components=n_pca_components,
                                    n_ica_components=n_ica_components,
                                    ccs_threshold=ccs_threshold,
                                    do_cca=do_cca,
                                    working_dir=working_dir,
                                    n_jobs=n_jobs)

    # Generate a list of pairs
    these_files = copy.copy(filenames)
    n_group1 = len(these_files)/2
    pairs = list()
    # XXX: Needs to be removed.
    random.seed(1)
    for _ in range(n_split_half):
        random.shuffle(these_files)
        pairs.append((these_files[:n_group1], these_files[n_group1:]))

    # Calculation correlation and reproducibility for each pair
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    correl = Parallel(n_jobs=n_jobs)(
                    delayed(memory.cache(canica_pair))(
                                            file_pair, n_pca_components, 
                                            mask, threshold,
                                            reference_icas,
                                            n_ica_components=n_ica_components,
                                            ccs_threshold=ccs_threshold,
                                            do_cca=do_cca,
                                            working_dir=working_dir,
                                            n_jobs=1, 
                                    )
                    for file_pair in pairs)

    un_thr_stats, thr_stats, reproducibility = zip(*correl)
    reproducibility = np.array(reproducibility).mean(axis=0)
    un_thr_stats = merge_stats(un_thr_stats)
    thr_stats = merge_stats(thr_stats)

    # Sort the ica maps by reproducibility
    order = np.argsort(reproducibility)[::-1]
    reproducibility = reproducibility[order]
    reference_icas = reference_icas[:, order]

    # Save a small report
    if working_dir is not None:
        report_file = file(os.path.join(working_dir,
                            'cross_validation_metrics.txt'), 'w')
        report_file.write("""

Unthresholded ICA maps
----------------------

%s

Thresholded ICA maps
----------------------

%s

""" % (report_stats(un_thr_stats),
       report_stats(thr_stats)))

    return reference_icas, mask, threshold, un_thr_stats, thr_stats, \
                        reproducibility

