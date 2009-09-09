"""
Apply CanICA on Sepideh's data.
"""
import os
from os.path import join as pjoin
import shutil

# Major neuroimaging libraries imports
import nifti

from joblib import PrintTime

# Local imports
from canica import canica, super_glob, save_ics

from canica.cross_validation import canica_split_half

#-------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------
N_JOBS = -1
SUBJECTS = range(1, 13)
N_PCA_COMPONENTS = 50
if 1:
 N_ICA_COMPONENTS = 42
 INPUT_GLOB = '~/data/data/subject%(subject)i/functional/fMRI/session1/swf*.img'
 WORKING_DIR = os.path.expanduser('~/data/canica/sepideh/')
else:
 INPUT_GLOB ='~/data/data-localizer/subject%(subject)i/swa*.img'
 WORKING_DIR = os.path.expanduser('~/data/canica/localizer/')
 N_ICA_COMPONENTS = 20


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

for do_cca, dirname in zip((True, False), ('cca', 'no_cca')):
    # XXX: Hard-coded cross_validation
    # Run CanICA
    icas, mask, threshold, un_thr_stats, thr_stats, reproducibility = \
                 canica_split_half(session_files, 
                                    n_split_half=20,
                                    n_pca_components=N_PCA_COMPONENTS,
                                    n_ica_components=N_ICA_COMPONENTS,
                                    do_cca=do_cca,
                                    working_dir=WORKING_DIR,
                                    n_jobs=N_JOBS)

    # And now output nifti and pretty pictures
    output_dir = pjoin(WORKING_DIR, dirname)
    titles = ['map % 2i, reproducibility: %.2f' % (index, r) 
                    for  index, r in enumerate(reproducibility)]
    save_ics(icas, mask, threshold, output_dir, fmri_header,
                    titles=titles)
    shutil.move(pjoin(WORKING_DIR, 'cross_validation_metrics.txt'),
                        output_dir)



print_time('Script finished', total=True)
# EOF ##########################################################################
