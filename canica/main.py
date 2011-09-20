"""
======================================================================
CanICA: Estimation of reproducible group-level ICA patterns for fMRI
======================================================================

"""
from os.path import join as pjoin
import itertools

# Major scientific libraries import
import numpy as np
from scipy import stats, linalg

# Neuroimaging libraries import
try:
    import nipy.labs.mask as mask_utils
except:
    import nipy.neurospin.utils.mask as mask_utils
try:
    from nibabel import load
except ImportError:
    from nipy.io.imageformats import load

# Unusual libraries import
from joblib import Memory, Parallel, delayed

# Local imports
from .algorithms.fastica import fastica
from .output import output


################################################################################
# First level analysis: principal component extraction at the subject level

def session_pca(raw_filenames, mask, smooth=False, 
                two_levels=False, n_first_components=None):
    """ Do the preprocessing and calculate the PCA components for a
        single session.

        Parameters
        -----------
        mask: 3d ndarray
            3D mask array: true where a voxel should be used.
        smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
        two_levels: boolean, optional
            If two_levels is True, the filenames are a list of filenames
            corresponding to various sessions, and a multi-level model is
            applied based on a first CCA.
        n_first_components: integer, optional
            The number of components to retain at the first level. This
            must be specified if two_levels is True.
    """
    # Data preprocessing and loading.
    if not two_levels:
        series, header = mask_utils.series_from_mask(raw_filenames, 
                                                        mask, smooth=smooth)

        # XXX: this should go in series_from_mask
        series -= series.mean(axis=-1)[:, np.newaxis]
        std = series.std(axis=-1)
        std[std==0] = 1
        series /= std[:, np.newaxis]
        del std
        # PCA
        components, loadings, _ = linalg.svd(series, full_matrices=False)
    else:
        if n_first_components is None:
            raise ValueError('If two_levels is True, n_first_components '
                              'must be specified')
        components = list()
        for session_files in raw_filenames:
            these_components, _, header = \
                            session_pca(session_files, mask, smooth=smooth)
            components.append(these_components[:,
                                :n_first_components])
            del these_components
        # CCA
        components = np.hstack(components)
        components, loadings, _ = linalg.svd(components,
                                            full_matrices=False)
    return components, loadings, header


def extract_subject_components(session_files, mask=None, n_jobs=1,
                                              two_levels=False,
                                              n_first_components=None,
                                              cachedir=None, smooth=False):
    """ Calculate principal components over different subjects.

        two_levels: boolean, optional
            If two_levels is True, the session_files passed represents
            multi-session data and a two-level hierarchical model is
            applied.
        n_first_components: integer, optional
            The number of components to retain at the first level. This
            must be specified if two_levels is True.
    """
    # If cachedir is None, the Memory object is transparent.
    memory = Memory(cachedir=cachedir, mmap_mode='r')
    cache = memory.cache

    # extract the common mask.
    if mask is None:
        if not two_levels:
            mask = cache(mask_utils.compute_mask_sessions)(session_files)
        else:
            mask = cache(mask_utils.compute_mask_sessions)(
                                            itertools.chain(*session_files))
    elif isinstance(mask, basestring):
        mask = load(mask).get_data().astype(np.bool)
 
    # Spread the load on multiple CPUs
    pca = delayed(cache(session_pca))
    session_pcas = Parallel(n_jobs=n_jobs)(
                                    pca(filenames, mask, smooth=smooth,
                                        two_levels=two_levels,
                                        n_first_components=n_first_components)
                                    for filenames in session_files)
    pcas, pca_loadings, headers = zip(*session_pcas)

    return pcas, mask, pca_loadings, headers[0]


################################################################################
# Group-level analysis: inter-subject extraction of ICA maps

def ica_step(group_maps, group_variance, cachedir=None):
    memory = Memory(cachedir=cachedir, mmap_mode='r')
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
                ccs_threshold=None, n_group_components=None, 
                cachedir=None):
    # Use asarray to cast to a non memmapped array
    subject_components = np.asarray(subject_components)

    # The group components (concatenated subject components)
    group_components = subject_components.T
    group_components = np.reshape(group_components,
                                    (group_components.shape[0], -1))
    # Save memory
    del subject_components

    # Inter-subject CCA
    memory = Memory(cachedir=cachedir, mmap_mode='r')
    svd = memory.cache(linalg.svd)
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
                n_ica_components=None, mask=None,
                two_levels=False,
                threshold_p_value=5e-2, n_jobs=1, working_dir=None, 
                return_mean=False, smooth=False,
                save_nifti=True, report=True):
    """ CanICA: reproducible multi-session ICA components from fMRI
        datasets.

        Parameters
        ----------
        filenames: list of list of strings.
            A list of list of filenames. The inner list gives the
            filenames of the datasets (nifti or analyze files) for one
            session, and the outer list is the list of the various
            sessions. See the super_glob to establish easily such a
            list.
        n_pca_components: int
            Number of principal components to use at the subject level.
        ccs_threshold: float, optional
            Threshold of the variance retained in the second level
            analysis.
        n_ica_components: float, optional
            Number of ICA to retain.
        mask: 3D boolean ndarray or string, optional
            The mask to use to extract the interesting time series.
            Can be either the array of the mask, or the name of a file
            containing the mask.
        threshold_p_value, float, optional
            The P value to use while thresholding the final ICA maps.
            See [Varoquaux ISBI 2010] for the description of the null 
            hypothesis.
        n_jobs: int, optional
            Number of jobs to start on a multi-processor machine.
            If -1, one job is started per CPU.
        two_levels: boolean, optional
            If two_levels is True, the filenames are a list of filenames
            corresponding to various sessions, and a multi-level model is
            applied based on a first CCA.
        working_dir: string, optional
            Optional directory name to use to store temporary cache.
        smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
        save_nifti: boolean, optional
            If save_nifti is True, a nifti file of the results is saved
            in the working_dir.
        report: boolean, optional
            If report is True, an html report is saved in the
            working_dir.

        Returns
        -------
        ica_maps: 2D ndarray
            The masked maps
        mask: 3D ndarray
            The corresponding mask
        threshold: float
            The threshold corresponding to the given p-value        
        header: dictonnary
            The header information of the input datasets
        group_components: 2D ndarray, optional
            If return_mean is True, the mean input image of each subject
            is returned.
  
        Notes
        -----
        Either n_ica_components of ccs_threshold should be specified, to
        indicate the final number of components.

        If report or save_nifti is specified, a working_dir should be
        specified.
    """
    if len(filenames) == 0:
        raise ValueError('No files where passed')
    if n_ica_components is None and ccs_threshold is None:
        raise ValueError('You need to specify either a number of '
            'ICA components, or a threshold for the canonical correlations')
    # Store the original mask value, for report
    orig_mask = mask

    if working_dir is not None:
        cachedir = pjoin(working_dir, 'cache')
    else:
        cachedir = None
    pcas, mask, variances, header = extract_subject_components(filenames,
                                            n_jobs=n_jobs,
                                            mask=mask,
                                            two_levels=two_levels,
                                            n_first_components=n_pca_components,
                                            cachedir=cachedir,
                                            smooth=smooth)
    # Get rid of memmapping in the header:
    if (hasattr(header, '_header_data') 
            and isinstance(header._header_data, np.ndarray)):
        header._header_data = np.asarray(header._header_data).copy()
    for key, value in header.items():
        if isinstance(value, np.ndarray):
            header[key] = np.asarray(value).copy()

    # Use np.asarray to get rid of memmapped arrays
    pcas = [np.asarray(pca.__array__())[:, :n_pca_components].T for pca in pcas]
    variances = [np.asarray(variance.__array__()[:n_pca_components])
                                            for variance in variances]

    group_components, group_variance = extract_group_components(pcas,
                                        variances, ccs_threshold=ccs_threshold,
                                        n_group_components=n_ica_components,
                                        cachedir=cachedir)

    ica_maps = ica_step(group_components, group_variance, cachedir=cachedir)

    threshold = (stats.norm.isf(0.5*threshold_p_value)
                                /np.sqrt(ica_maps.shape[0]))

    if save_nifti or report:
        parameters = dict(
                filenames=filenames,
                n_pca_components=n_pca_components,
                ccs_threshold=ccs_threshold,
                n_ica_components=n_ica_components,
                mask=orig_mask,
                threshold_p_value=threshold_p_value,
                smooth=smooth,
                working_dir=working_dir,
            )
        output(ica_maps, mask, threshold, header, working_dir=working_dir,
                parameters=parameters,
                mean=group_components.T[0], 
                save_nifti=save_nifti, report=report)

    if not return_mean:
        return ica_maps, mask, threshold, header
    else:
        return ica_maps, mask, threshold, header, group_components.T[0]

