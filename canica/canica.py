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
    series = series_from_mask([raw_filenames, ], mask)

    # PCA
    components, loadings, _ = np.linalg.svd(series, full_matrices=False)

    return components, loadings


def intra_subject_pcas(session_files, mask=None, n_jobs=1,
                            working_dir=None):
    """ Calculate principal components over different subjects.
    """
    # If working_dir is None, the Memory object is transparent.
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    cache = memory.cache

    # extract the common mask. 
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

def ica_after_cca(pcas, ccs_threshold=None, working_dir=None,
                n_ica_components=None, cca=True):
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    svd = memory.cache(np.linalg.svd)
    cca_maps, ccs, _ = svd(pcas, full_matrices=False)
    if n_ica_components is None:
        n_ica_components = np.argmin(ccs > ccs_threshold)
    cca_maps = cca_maps[:, :n_ica_components]

    # We do a spatial ICA: the arrays are transposed in the following, 
    # axis1 = component, and axis2 is voxel number.
    _, ica_maps = memory.cache(fastica)(cca_maps.T, 
                                           n_ica_components, whiten=False)

    # Project the ICAs on the CCA maps to give a 'cross-subject
    # reproducibility' score.
    proj = np.dot(ica_maps, cca_maps[:, :n_ica_components])
    reproducibility_score = (np.abs(proj)*ccs[:n_ica_components]).sum(axis=-1)

    order = np.argsort(reproducibility_score)[::-1]

    ica_maps = ica_maps[order, :]

    return ica_maps.T

################################################################################
# Thresholding and post-processing

################################################################################
# Actual estimation of the complete CanICA model

def canica(filenames, n_pca_components, ccs_threshold=None,
                n_ica_components=None, cca=True,
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
        cca: boolean, optional
            If True, a canonical correlations analysis (CCA) is used to
            go from the subject patterns to the group model.
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
    # First level analysis
    pcas, mask, variances = intra_subject_pcas(filenames, n_jobs=n_jobs, 
                                        working_dir=working_dir)

    if not cca:
        for pca, variance in zip(pcas, variances):
            pca *= variance

    # The group principal components (concatenated subject PCs)
    # Use asarray to cast to a non memmapped array
    pcas = np.asarray([pca[:, :n_pca_components].T for pca in pcas]).T

    pcas = np.reshape(pcas, (pcas.shape[0], -1))
    # Inter-subject CCA and ICA 
    ica_maps = ica_after_cca(pcas, ccs_threshold=ccs_threshold,
                                      n_ica_components=n_ica_components,
                                      working_dir=working_dir)

    threshold = stats.norm.isf(0.5*threshold_p_value)/np.sqrt(icas.shape[0])

    return ica_maps, mask, threshold

