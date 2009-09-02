"""
Apply CanICA on Sepideh's data.
"""
import os

# Major scientific libraries imports
import numpy as np
from scipy import stats

# Major neuroimaging libraries imports
import nifti

from joblib import Memory, PrintTime

# Local imports
from canica import canica
# For interactive work
reload(canica)
from canica.tools.super_glob import super_glob

# FIXME: I need to merge joblib's logger and CanICA's logger.

WORKING_DIR = 'canica_maps'

cached_super_glob = Memory(cachedir=WORKING_DIR).cache(super_glob)

#-------------------------------------------------------------------------
# Disk IO
#-------------------------------------------------------------------------

session_files = super_glob(
    os.path.expanduser(
            '~/data/data/subject%(subject)i/functional/fMRI/session1/sw*.hdr'
    ),
    subject=range(1, 13),
  )

print_time = PrintTime()
#-------------------------------------------------------------------------
# CanICA estimation
#-------------------------------------------------------------------------
# XXX: Absolute hack in the ccs_threshold
common_icas, mask = canica.canica(session_files, 
                                  n_pca_components=50,
                                  ccs_threshold=0.14*len(session_files),
                                  working_dir=WORKING_DIR,
                                  n_jobs=-1)

print_time('Computing common ICA decomposition')

#-------------------------------------------------------------------------
# Save output maps to nifti
#-------------------------------------------------------------------------

# First retrieve the header of the original data.
fmri_header = nifti.NiftiImage(session_files[0][0]).header
sform = fmri_header['sform']

from fff2.viz import activation_maps as am
import pylab as pl

def auto_sign(map, threshold=0):
    positiv_mass = (map > threshold).sum()
    negativ_mass = (map < threshold).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1

# Simple heuristic for thresholding
p_value = 5e-3
threshold = stats.norm.isf(0.5*p_value)/np.sqrt(common_icas.shape[0])

# put in order n_ic, n_voxels
common_icas = common_icas.T

if 1:
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Create an MPL colormap to suit my needs:
    import matplotlib as mp
    cdict = {'blue': ((0., 1.0, 1.0), 
                        (0.1825, 1.0, 1.0), 
                        (0.5, 0.2, 0.0), 
                        (0.873, 0.0, 0.0), 
                        (1.0, 1.0, 1.0)),
            'green': ((0.0, 1.0, 1.0),
                        (0.1825, 0.5, 0.5),
                        (0.5, 0.1, 0.0),
                        (0.6825, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
            'red'  : ((0.0, 1.0, 1.0),
                        (0.3873, .0, .0),
                        (0.5, 0., 0.15),
                        (0.6825, 1.0, 1.0),
                        (1.0, 1.0, 1.0))
            }

    cmap = mp.colors.LinearSegmentedColormap('my_colormap', cdict, 512)

    for index, ic in enumerate(common_icas):
        print 'Outputing map %i out of %i' % (index, len(common_icas)) 
        this_map = np.zeros(mask.shape)
        this_map[mask] = ic
        # Modify the 3D map rather than the IC to avoid modifying
        # the original data.
        this_map *= auto_sign(ic, threshold=threshold)
        this_header = fmri_header.copy()
        this_header['intent_name'] = 'IC'
        nifti.NiftiImage(this_map.T, this_header).save(
                                WORKING_DIR + '/ic_no_thr%02i.nii' % index
                            )
        
        this_map[this_map < threshold] = 0
        nifti.NiftiImage(this_map.T, this_header).save(
                                WORKING_DIR + 'ic%02i.nii' % index
                            )
        x, y, z = am.find_cut_coords(this_map, mask=mask,
                                        activation_threshold=1e-10)
        # XXX: This is all wrong: we are struggling, jumping from voxel to
        # Talairach.
        y, x, z = am.coord_transform(x, y, z, sform)
        if np.any(this_map != 0):
            # Ugly trick to force the colormap to be symetric:
            this_map[0, 0, 0] = -this_map.max()
            this_map = np.ma.masked_equal(this_map, 0, copy=False)
            am.plot_map_2d(this_map, sform, (x, y, z), figure_num=512,
                        cmap=cmap)
        else:
            pl.clf()
        pl.savefig(WORKING_DIR + '/map_%02i.png' % index)



print_time('Script finished', total=True)
# EOF ##########################################################################
