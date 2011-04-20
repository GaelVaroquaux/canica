"""
Apply CanICA on sample data
"""
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org> 
# License: BSD Style.

# Local imports
from canica import canica, canica_split_half

#-------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------
N_JOBS = -1
SUBJECTS = range(1, 13)
N_PCA_COMPONENTS = 50
THRESHOLD_P_VALUE = 20e-2
N_ICA_COMPONENTS = 42
INPUT_GLOB = '/volatile/varoquau/data/flore/subject%i_session1.nii'
WORKING_DIR = '/tmp/data/canica/'

#-------------------------------------------------------------------------
# Disk IO
#-------------------------------------------------------------------------
# The super_glob returns a list of globs for each value of subject
session_files = [INPUT_GLOB % i for i in SUBJECTS]

#-------------------------------------------------------------------------
# CanICA estimation
#-------------------------------------------------------------------------

if 1:
    # Simply run CanICA
    icas, mask, threshold, header = \
                canica(session_files, 
                                  smooth=2,
                                  n_pca_components=N_PCA_COMPONENTS,
                                  n_ica_components=N_ICA_COMPONENTS,
                                  threshold_p_value=THRESHOLD_P_VALUE,
                                  working_dir=WORKING_DIR,
                                  n_jobs=N_JOBS,
                                  report=True)
else:
    # Run CanICA with a split-half cross-validation study
    icas, mask, threshold, un_thr_stats, thr_stats, header, reproducibility = \
                canica_split_half(session_files, 
                                  n_split_half=20,
                                  smooth=2,
                                  n_pca_components=N_PCA_COMPONENTS,
                                  n_ica_components=N_ICA_COMPONENTS,
                                  threshold_p_value=THRESHOLD_P_VALUE,
                                  working_dir=WORKING_DIR,
                                  n_jobs=N_JOBS,
                                  report=True)


# EOF ##########################################################################
