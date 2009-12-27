"""
Functions to read series and masks on the raw data.
"""

# Major scientific libraries import
import numpy as np

# Neuroimaging library imports
from nipy.io.imageformats import load

################################################################################
# Series and mask extraction 
################################################################################

def series_from_mask(session_files, mask, dtype=np.float32):
    """ Read the time series from the given sessions filenames, using the mask.

        Parameters
        -----------
        session_files: list of list of nifti file names. 
            Files are grouped by session.
        mask: 3d ndarray
            3D mask array: true where a voxel should be used.
        
        Returns
        --------
        session_series: ndarray
            3D array of time course: (session, voxel, time)
        header: header object
            The header of the first file.
    """
    # XXX: What if the file lengths do not match!
    mask = mask.astype(np.bool)
    nb_time_points = len(session_files[0])
    if len(session_files[0]) == 1:
        # We have a 4D nifti file
        nb_time_points = load(session_files[0][0]).get_data().shape[-1]
    session_series = np.zeros((len(session_files), mask.sum(), 
                                            nb_time_points),
                                    dtype=dtype)

    for session_index, filenames in enumerate(session_files):
        if len(filenames) == 1:
            # We have a 4D nifti file
            data_file = load(filenames[0])
            data = data_file.get_data()
            session_series[session_index, :, :] = data[mask].astype(dtype)
            if not 'header' in locals():
                header = data_file.get_header()
            # Free memory early
            del data, data_file
        else:
            for file_index, filename in enumerate(filenames):
                data_file = load(filename)
                data = data_file.get_data()
                    
                session_series[session_index, :, file_index] = \
                                data[mask].astype(np.float32)
                # Free memory early
                if not 'header' in locals():
                    header = data_file.get_header()
                del data

    return session_series.squeeze(), header


