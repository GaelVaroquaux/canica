"""
Functions to read series and masks on the raw data.
"""

# Major scientific libraries import
import numpy as np
from scipy.signal import detrend

# Neuroimaging library imports
import nifti
from fff2.utils.mask import compute_mask_sessions
from fff2.utils.mask import compute_mask_intra

# Local imports
from io import get_fmri_session_files, get_grey_matter_masks, FileGlober


def read_grey_matter_mask(subjects, brain_mask, fmri_header,
                                                io_context=None):
    """ Load a common grey matter mask between subjects.

    INPUT:
    - subjects: list of subjects number
    - brain_mask: 3D array: common brain mask, as found by fff2 mask utils.
    - fmri_header: header of the nifti images from which the brain masks have 
                   been calculated.

    OUTPUT:
    - mask: 3D mask array.
    """
    if io_context is not None and 'grey_matter_pattern' in io_context:
        pattern = io_context['grey_matter_pattern']
        file_getter = FileGlober(pattern)
    else:
        file_getter = get_grey_matter_masks
    grey_matter_mask = np.zeros(brain_mask.shape, np.int8)

    for filename in file_getter(subject=subjects, io_context=io_context):
        filename = filename[0]
        from scipy.ndimage import affine_transform
        grey_matter_img = nifti.NiftiImage(filename)
        grey_matter_header = grey_matter_img.header
        grey_matter_img = grey_matter_img.data
        affine = np.dot(np.linalg.inv(fmri_header['sform']),
                                       grey_matter_header['sform'])
        offset = affine[:3, -1]
        affine = affine[:3, :3]
        grey_matter_img = affine_transform(grey_matter_img,
                                            np.linalg.inv(affine),
                                            offset=-offset,
                                            output_shape=brain_mask.T.shape,
                                            order=0,
                                            mode='nearest').T
        grey_matter_img -= grey_matter_img.min()
        grey_matter_mask += (grey_matter_img > grey_matter_img.max()*0.3)


    grey_matter_mask = (grey_matter_mask > len(subjects)*0.2)
    # Take only the voxels in the whole brain mask found using the clever
    # algorithm, and the grey matter mask.
    return grey_matter_mask * brain_mask


def read_series(session_files, mask):
    """ Read the time series from the given sessions filenames, using the mask.

        INPUT:
        - session_files: list of list of nifti file name. Files are grouped by 
          session.
        - mask: 3D mask array.
        OUTPUT:
        - 3D array of time course: (session, voxel, time)
    """
    mask = mask.astype(np.bool)
    session_series = np.zeros((len(session_files), mask.sum(), 
                                            len(session_files[0])),
                                    np.float32)
    for session_index, filenames in enumerate(session_files):
        this_session = list()
        for file_index, filename in enumerate(filenames):
            data = nifti.NiftiImage(filename).asarray().T
            session_series[session_index, :, file_index] = \
                            data[mask].astype(np.float32)

    return session_series.squeeze()


def read_masks(subjects, sessions, io_context=None, threshold=0.5, cc=1):
    """ Read the whole brain mask, grey matter mask.

        INPUT:
        - subjects: list of subject numbers
        - sessions: list of session numbers
        - io_context: optionnal dictionnary specifying the upstream
          data directory, and the local data directory, if caching is
          desired.

        OUTPUT:
        - brain_mask: 3D mask array of the whole brain
        - grey_matter_mask: 3D mask array of the grey matter
    """
    if io_context is not None and 'fmri_data_pattern' in io_context:
        pattern = io_context['fmri_data_pattern']
        file_getter = FileGlober(pattern)
    else:
        file_getter = get_fmri_session_files

    sessions_files = file_getter(subject=subjects,
                                 session=sessions,
                                 io_context=io_context)

    brain_mask = compute_mask_sessions(sessions_files,
                                       threshold=threshold)
    fmri_header = nifti.NiftiImage(sessions_files[0][0]).header

    grey_matter_mask = read_grey_matter_mask(subjects, brain_mask, 
                                             fmri_header,
                                             io_context=io_context)
    return brain_mask, grey_matter_mask, fmri_header


def read_series_and_mask(subjects, sessions, io_context=None):
    """ Read the whole brain mask and time courses.

        INPUT:
        - subjects: list of subject numbers
        - sessions: list of session numbers
        - io_context: optionnal dictionnary specifying the upstream
          data directory, and the local data directory, if caching is
          desired.

        OUTPUT:
        - brain_mask: 3D mask array of the whole brain
        - series: List of 2D arrays giving for each subject the time
          courses in the grey matter mask.
        - fmri_header: Nifti header corresponding to the frmi images.
    """
    sessions_files = get_fmri_session_files(subject=subjects,
                                            session=sessions,
                                            io_context=io_context)

    brain_mask = compute_mask_sessions(sessions_files, threshold=0.5)
    fmri_header = nifti.NiftiImage(sessions_files[0][0]).header

    series = read_series(sessions_files, brain_mask)

    return brain_mask, series.squeeze(), fmri_header


################################################################################
# Data preprocessing and loading.
################################################################################

def load_session(subject, session, io_context=None, clean=False):
    """ Read the whole brain mask and time courses for a session.

        INPUT:
        - subject: subject number
        - session: session number
        - io_context: optionnal dictionnary specifying the upstream
          data directory, and the local data directory, if caching is
          desired.

        OUTPUT:
        - brain_mask: 3D mask array of the whole brain
        - series: 2D array giving the time courses in the brain matter mask.
        - fmri_header: Nifti header corresponding to the frmi images.
        - mean: 3D array of the mean of the images.
        - logvar: 3D array of the logvariance of the time courses in the 
                  series in the mask of the brain.
    """
    sessions_files = get_fmri_session_files(subject=subject,
                                                session=session,
                                                io_context=io_context)

    fmri_header = nifti.NiftiImage(sessions_files[0]).header
    brain_mask, mean = compute_mask_intra(sessions_files, None,
                                return_mean=True)
    # Extract the data of the nifti image.
    brain_mask = brain_mask.asarray().T
    # The mean is ordered in nifti orientation.
    mean = mean.T
    series = read_series([sessions_files, ], brain_mask)

    # Compute log-variance map
    logvar = np.log(series.var(axis=-1))
    logvar_map = np.zeros(brain_mask.shape, np.float32)
    logvar_map[brain_mask.astype(np.bool)] = logvar

    if clean:
        # Detrend a renormalize the map
        for serie in series:
            serie[:] = detrend(serie)
            #########################################################
            ## Lowpass filter
            ## We have a TR of 1.5s
            #freq = fftfreq(series.shape[-1], 1.5)
            #f = fft(serie)
            ## Cut at 0.1 Hz
            #f[freq>0.1] = 0
            #serie = ifft(f)

            # Normalize the variance
            std = np.std(serie)
            if std == 0:
                std = 1
            serie /= std

    return brain_mask, series, fmri_header, logvar_map, mean


def read_series_and_gm_masks(subjects, sessions, io_context=None):
    """ Read the whole brain mask, grey matter mask and time courses,
        restricted in the grey matter mask.

        INPUT:
        - subjects: list of subject numbers
        - sessions: list of session numbers
        - io_context: optionnal dictionnary specifying the upstream
          data directory, and the local data directory, if caching is
          desired.

        OUTPUT:
        - brain_mask: 3D mask array of the whole brain
        - grey_matter_mask: 3D mask array of the grey matter
        - series: List of 2D arrays giving for each subject the time
          courses in the grey matter mask.
        - fmri_header: Nifti header corresponding to the frmi images.
    """
    sessions_files = get_fmri_session_files(subject=subjects,
                                            session=sessions,
                                            io_context=io_context)

    brain_mask = compute_mask_sessions(sessions_files, threshold=0.5)
    fmri_header = nifti.NiftiImage(sessions_files[0][0]).header

    grey_matter_mask = read_grey_matter_mask(subjects, brain_mask, 
                                             fmri_header,
                                             io_context=io_context)

    series = read_series(sessions_files, grey_matter_mask)

    return brain_mask, grey_matter_mask, series.squeeze(), fmri_header

