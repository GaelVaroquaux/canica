"""
======================================================================
CanICA: Estimation of reproducible group-level ICA patterns for fMRI
======================================================================

"""

# Major scientific libraries import
import numpy as np

# Neuroimaging libraries import
import nipy.neurospin.utils.mask as mask_utils

# Local imports
from .tools.parallel import PMap
from .io import series_from_mask
from .algorithms.fastica import fastica


################################################################################
# First level analysis: principal component extraction at the subject level
################################################################################

def session_pca(raw_filenames, mask, n_pca_components=75):
    """ Do the preprocessing and calculate the PCA components for a
        single session.
    """
    # FIXME: No logging facility.

    # Data preprocessing and loading.
    series = series_from_mask([raw_filenames, ], mask)

    # PCA
    components, loadings, _ = np.linalg.svd(series, full_matrices=False)
    
    return components, loadings


def intra_subject_pcas(session_files, mask=None, n_jobs=1, n_pca_components=75):
    """ Calculate principal components over different subjects.
    """
    if mask is None:
        # extract the common mask. We have to transpose because
        # nipy.neurospin does not transpose, whereas we do.
        mask = mask_utils.compute_mask_sessions(session_files).T
    # Do the intra-subject PCAs
    session_pca_output = PMap(n_jobs=n_jobs)(
                        session_pca, session_files, 
                        mask=mask,
                        n_pca_components=n_pca_components,
                        )
    pcas, pca_loadings = zip(*session_pca_output)

    return pcas, mask, pca_loadings



################################################################################
# Group-level analysis: inter-subject extraction of ICA maps
################################################################################

def ica_after_cca(pcas, ccs_threshold=1.6):
    cca_maps, ccs, _ = np.linalg.svd(pcas, full_matrices=False)
    n_cca_components = np.argmin(ccs > ccs_threshold)
    _, common_icas = fastica(cca_maps[:, :n_cca_components], 
                                n_cca_components, whiten=False)
    common_icas = common_icas.T
    # Normalize to 1 the ICA maps:
    normed_icas = common_icas/float(cca_maps.shape[0])

    # Project the ICAs on the CCA maps to give a 'cross-subject
    # reproducibility' score.
    proj = np.dot(normed_icas, cca_maps[:, :n_cca_components])
    reproducibility_score = (np.abs(proj)*ccs[:n_cca_components]).sum(axis=-1)

    order = np.argsort(reproducibility_score)[::-1]

    common_icas = common_icas[order, :]

    return common_icas



################################################################################
# Actual estimation of the complete CanICA model
################################################################################
def canica(filenames, n_pca_components, ccs_threshold, n_jobs=1):
    # FIXME: No logging facility.

    # First level analysis
    pcas, mask, _ = intra_subject_pcas(filenames, n_jobs=1)

    # Inter-subject CCA and ICA 
    common_icas = ica_after_cca(pcas, ccs_threshold=ccs_threshold)

    return common_icas, mask

