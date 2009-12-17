"""
Apply CanICA on sample data
"""
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org> 
# License: BSD Style.


# Major neuroimaging libraries imports
from nipy.io.imageformats import load

# Local imports
from canica import super_glob, save_ics
from canica.cross_validation import canica_split_half

#-------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------
N_JOBS = -1
SUBJECTS = range(1, 13)
N_PCA_COMPONENTS = 50
THRESHOLD_P_VALUE = 20e-2
N_ICA_COMPONENTS = 42
INPUT_GLOB = '/volatile/varoquau/data/data/subject%(subject)i/functional/fMRI/session1/swf*.img'
WORKING_DIR = '/tmp/data/canica/'


#-------------------------------------------------------------------------
# Disk IO
#-------------------------------------------------------------------------
# The super_glob returns a list of globs for each value of subject
session_files = super_glob(INPUT_GLOB, subject=SUBJECTS)

# Retrieve the header of the original data, to be able to save with
# the same information
fmri_header = load(session_files[0][0]).get_header()

#-------------------------------------------------------------------------
# CanICA estimation
#-------------------------------------------------------------------------

# Run CanICA with a split-half cross-validation study
icas, mask, threshold, un_thr_stats, thr_stats, reproducibility = \
                canica_split_half(session_files, 
                                  n_split_half=20,
                                  n_pca_components=N_PCA_COMPONENTS,
                                  n_ica_components=N_ICA_COMPONENTS,
                                  threshold_p_value=THRESHOLD_P_VALUE,
                                  working_dir=WORKING_DIR,
                                  n_jobs=N_JOBS)


# And now output nifti and pretty pictures
titles = ['map % 2i, reproducibility: %.2f' % (index, r) 
                for  index, r in enumerate(reproducibility)]
save_ics(icas, mask, threshold, WORKING_DIR, fmri_header,
                titles=titles, format='pdf')


# EOF ##########################################################################
