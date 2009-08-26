"""
Apply CanICA on Sepideh's data.
"""
import os

# Major scientific libraries imports
import numpy as np
from scipy import stats

# Major neuroimaging libraries imports
import nifti

# Local imports
from canica import canica
from canica.tools.super_glob import super_glob
from canica.tools.logger import PrintTime

#-------------------------------------------------------------------------
# Disk IO
#-------------------------------------------------------------------------

session_files = super_glob(
    os.path.expanduser(
  '~/data/data/subject%(subject)i/functional/fMRI/session1/sw*.hdr'
    ),
    subject=range(1, 12),
  )

print_time = PrintTime()
#-------------------------------------------------------------------------
# CanICA estimation
#-------------------------------------------------------------------------
common_icas, mask = canica.canica(session_files) 

print_time('Computing common ICA decomposition')

#-------------------------------------------------------------------------
# Save output maps to nifti
#-------------------------------------------------------------------------
fmri_header = nifti.NiftiImage(session_files[0][0]).header
sform = fmri_header['sform']

from fff2.viz import activation_maps as am
import pylab as pl

def auto_sign(map):
    positiv_mass = (map > 0).sum()
    negativ_mass = (map < 0).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1

# Simple heuristic for thresholding
p_value = 5e-3
threshold = stats.norm.isf(0.5*p_value)/np.sqrt(common_icas.shape[1])
common_icas_no_thr = common_icas
common_icas = common_icas.copy()
common_icas[np.abs(common_icas) < threshold] = 0

outdir = 'canica_maps' 
if not os.path.exists(outdir):
    os.mkdir(outdir)

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

cmap = mp.colors.LinearSegmentedColormap('my_colormap',cdict,512)


for index, (ic, ic_no_thr) in enumerate(
            zip(common_icas, common_icas_no_thr)):
    print 'Outputing map %i out of %i' % (index, len(common_icas)) 
    this_map = np.zeros(mask.shape)
    this_map[mask] = ic_no_thr
    this_map *= auto_sign(this_map)
    this_header = fmri_header.copy()
    this_header['intent_name'] = 'IC'
    nifti.NiftiImage(this_map.T, this_header).save(
                            outdir + 'ic_no_thr%02i.nii' % index
                        )
    
    this_map[mask] = ic
    this_map *= auto_sign(this_map)
    this_header = fmri_header.copy()
    this_header['intent_name'] = 'IC'
    nifti.NiftiImage(this_map.T, this_header).save(
                            outdir + 'ic%02i.nii' % index
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
    pl.savefig(outdir + '/map_%02i.png' % index)



print_time('Script finished')
# EOF ##########################################################################
