"""
======================================================================
CanICA: Estimation of reproducible group-level ICA patterns for fMRI
======================================================================

"""

# Major scientific libraries import
import numpy as np

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

    # extract the common mask. We have to transpose because
    # nipy.neurospin does not transpose, whereas we do.
    mask = cache(mask_utils.compute_mask_sessions)(session_files).T

    # Spread the load on multiple CPUs
    pca = delayed(cache(session_pca))
    session_pcas = Parallel(n_jobs=n_jobs)( 
                                    pca(filenames, mask)
                                    for filenames in session_files)
    pcas, pca_loadings = zip(*session_pcas)

    return pcas, mask, pca_loadings


################################################################################
# Group-level analysis: inter-subject extraction of ICA maps

def ica_after_cca(pcas, ccs_threshold=1.6, working_dir=None):
    memory = Memory(cachedir=working_dir, debug=True, mmap_mode='r')
    svd = memory.cache(np.linalg.svd)
    cca_maps, ccs, _ = svd(pcas, full_matrices=False)
    n_cca_components = np.argmin(ccs > ccs_threshold)
    cca_maps = cca_maps[:, :n_cca_components]

    # We do a spatial ICA: the arrays are transposed in the following, 
    # axis1 = component, and axis2 is voxel number.
    _, common_icas = memory.cache(fastica)(cca_maps.T, 
                                           n_cca_components, whiten=False)

    # Project the ICAs on the CCA maps to give a 'cross-subject
    # reproducibility' score.
    proj = np.dot(common_icas, cca_maps[:, :n_cca_components])
    reproducibility_score = (np.abs(proj)*ccs[:n_cca_components]).sum(axis=-1)

    order = np.argsort(reproducibility_score)[::-1]

    common_icas = common_icas[order, :]

    return common_icas.T

################################################################################
# Thresholding and post-processing

################################################################################
# Actual estimation of the complete CanICA model

def canica(filenames, n_pca_components, ccs_threshold, n_jobs=1, 
                                working_dir=None):
    """ CanICA
    """
    # First level analysis
    pcas, mask, _ = intra_subject_pcas(filenames, n_jobs=n_jobs, 
                                        working_dir=working_dir)

    # The group principal components (concatenated subject PCs)
    # Use asarray to cast to a non memmapped array
    pcas = np.asarray([pca[:, :n_pca_components].T for pca in pcas]).T

    pcas = np.reshape(pcas, (pcas.shape[0], -1))
    # Inter-subject CCA and ICA 
    common_icas = ica_after_cca(pcas, ccs_threshold=ccs_threshold,
                                            working_dir=working_dir)

    return common_icas, mask

