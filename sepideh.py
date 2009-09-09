"""
Apply CanICA on Sepideh's data.
"""
import os
from os.path import join as pjoin

# Major neuroimaging libraries imports
import nifti

from joblib import PrintTime

# Local imports
from canica import canica, super_glob, save_ics

#-------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------
N_JOBS = -1
SUBJECTS = range(1, 13)
N_PCA_COMPONENTS = 50
N_ICA_COMPONENTS = 42
if 0:
 INPUT_GLOB = '~/data/data/subject%(subject)i/functional/fMRI/session1/swf*.img'
 WORKING_DIR = os.path.expanduser('~/data/canica/sepideh/')
else:
 INPUT_GLOB ='~/data/data-localizer/subject%(subject)i/swa*.img'
 WORKING_DIR = os.path.expanduser('~/data/canica/localizer/')


#-------------------------------------------------------------------------
# Disk IO
#-------------------------------------------------------------------------
print_time = PrintTime()

session_files = super_glob(INPUT_GLOB, subject=SUBJECTS)

# Retrieve the header of the original data.
fmri_header = nifti.NiftiImage(session_files[0][0]).header

#-------------------------------------------------------------------------
# CanICA estimation
#-------------------------------------------------------------------------

for cca, dirname in zip((True, False), ('cca', 'no_cca')):
    # Run CanICA
    icas, mask, threshold = canica(session_files, 
                                    n_pca_components=N_PCA_COMPONENTS,
                                    n_ica_components=N_ICA_COMPONENTS,
                                    cca=cca,
                                    working_dir=WORKING_DIR,
                                    n_jobs=N_JOBS)

    # And now output nifti and pretty pictures
    output_dir = pjoin(WORKING_DIR, dirname)
    save_ics(icas, mask, threshold, output_dir, fmri_header)


print_time('Script finished', total=True)
# EOF ##########################################################################
