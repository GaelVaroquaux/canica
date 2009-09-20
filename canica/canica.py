"""
======================================================================
CanICA: Estimation of reproducible group-level ICA patterns for fMRI
======================================================================

"""

# Major scientific libraries import
import numpy as np
from scipy import stats

# Neuroimaging libraries import
import nipy.neurospin.utils.mask as mask_utils

# Unusual libraries import
from joblib import Memory

# Local imports
from .tools.parallel import Parallel, delayed
from .io import series_from_mask
from .algorithms.fastica import fastica


################################################################################
# First level analysis: principal component extraction at the subject level

def session_pca(raw_filenames, mask):
    """ Do the preprocessing and calculate the PCA components for a
        single session.
    """
    # Data preprocessing and loading.
    # XXX: series_from_mask should go in nipy.neurospin.utils.mask
    series = series_from_mask([raw_filenames, ], mask)

    # XXX: this should go in series_from_mask
    series -= series.mean(axis=-1)[:, np.newaxis]
    std = series.std(axis=-1)
    std[std==0] = 1
    series /= std[:, np.newaxis]
    del std
    # PCA
    components, loadings, _ = np.linalg.svd(series, full_matrices=False)

    return components, loadings


def extract_subject_components(session_files, mask=None, n_jobs=1,
                                              working_dir=None):
    """ Calculate principal components over different subjects.
    """
    # If working_dir is None, the Memory object is transparent.
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    cache = memory.cache

    # extract the common mask. 
    if mask is None:
        mask = cache(mask_utils.compute_mask_sessions)(session_files)

    # Spread the load on multiple CPUs
    pca = delayed(cache(session_pca))
    session_pcas = Parallel(n_jobs=n_jobs)( 
                                    pca(filenames, mask)
                                    for filenames in session_files)
    pcas, pca_loadings = zip(*session_pcas)

    return pcas, mask, pca_loadings


################################################################################
# Group-level analysis: inter-subject extraction of ICA maps

def ica_step(group_maps, group_variance, working_dir=None):
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    # We do a spatial ICA: the arrays are transposed in the following, 
    # axis1 = component, and axis2 is voxel number.
    _, ica_maps = memory.cache(fastica)(group_maps.T, whiten=False)

    # Project the ICAs on the group maps to give a 'cross-subject
    # reproducibility' score.
    proj = np.dot(ica_maps, group_maps)
    reproducibility_score = (np.abs(proj)*group_variance).sum(axis=-1)

    order = np.argsort(reproducibility_score)[::-1]

    ica_maps = ica_maps[order, :]

    return ica_maps.T


def extract_group_components(subject_components, variances, 
                ccs_threshold=None, n_group_components=None, do_cca=True,
                working_dir=None):
    # Use asarray to cast to a non memmapped array
    subject_components = np.asarray(subject_components)

    if not do_cca:
        for component, variance in zip(subject_components, variances):
            component *= variance[:, np.newaxis]
        del component, variance

    # The group components (concatenated subject components)
    group_components = subject_components.T
    group_components = np.reshape(group_components, 
                                    (group_components.shape[0], -1))
    # Save memory
    del subject_components

    # Inter-subject CCA 
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    svd = memory.cache(np.linalg.svd)
    cca_maps, ccs, _ = svd(group_components, full_matrices=False)
    # Save memory
    del group_components
    if n_group_components is None:
        n_group_components = np.argmin(ccs > ccs_threshold)
    cca_maps = cca_maps[:, :n_group_components]
    ccs = ccs[:n_group_components]
    return cca_maps, ccs


################################################################################
# Actual estimation of the complete CanICA model

def canica(filenames, n_pca_components, ccs_threshold=None,
                n_ica_components=None, do_cca=True, mask=None,
                threshold_p_value=5e-3,
                n_jobs=1, working_dir=None):
    """ CanICA

        Parameters
        ----------
        n_pca_components: int
            Number of principal components to use at the subject level.
        ccs_threshold: float, optional
            Threshold of the variance retained in the second level
            analysis.
        n_ica_components: float, optional
            Number of ICA to retain.
        do_cca: boolean, optional
            If True, a canonical correlations analysis (CCA) is used to
            go from the subject patterns to the group model.
        mask: 3D boolean ndarray, optional
            The mask to use to extract the interesting time series.
        threshold_p_value, float, optional
            The P value to use while thresholding the final ICA maps.
        n_jobs: int, optional
            Number of jobs to start on a multi-processor machine.
            If -1, one job is started per CPU.
        working_dir: string, optional
            Optional directory name to use to store temporary cache.

        Notes
        -----
        Either n_ica_components of ccs_threshold should be specified, to 
        indicate the final number of components.
    """
    if n_ica_components is None and ccs_threshold is None:
        raise ValueError('You need to specify either a number of '
            'ICA components, or a threshold for the canonical correlations')
    
    pcas, mask, variances = extract_subject_components(filenames, 
                                                       n_jobs=n_jobs, 
                                                       mask=mask,
                                                       working_dir=working_dir)

    # Use np.asarray to get rid of memmapped arrays
    pcas = [np.asarray(pca[:, :n_pca_components].T) for pca in pcas]
    variances = [np.asarray(variance[:n_pca_components]) 
                                            for variance in variances]

    group_components, group_variance = extract_group_components(pcas, 
                                        variances, ccs_threshold=ccs_threshold, 
                                        n_group_components=n_ica_components, 
                                        do_cca=do_cca, working_dir=working_dir)

    ica_maps = ica_step(group_components, group_variance,
                                          working_dir=working_dir)

    threshold = (stats.norm.isf(0.5*threshold_p_value)
                                /np.sqrt(ica_maps.shape[0]))

    return ica_maps, mask, threshold


