"""
Visualization functions for CanICA.
"""
import os
from os.path import join as pjoin

# Major scientific library imports
import numpy as np
import pylab as pl
import matplotlib as mp

# Neuroimaging library imports
import nifti

from fff2.viz import activation_maps as am

# Create an MPL colormap to suit my needs:
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


def auto_sign(map, threshold=0):
    positiv_mass = (map > threshold).sum()
    negativ_mass = (map < -threshold).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1


def save_ics(icas, mask, threshold, output_dir, header, titles=None):
    # put in order n_ic, n_voxels
    icas = icas.T

    header['intent_name'] = 'IC'
    sform = header['sform']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for index, ic in enumerate(icas):
        print 'Outputing map %i out of %i' % (index + 1, len(icas)) 
        map3d = np.zeros(mask.shape)
        map3d[mask] = ic
        # Modify the 3D map rather than the IC to avoid modifying
        # the original data.
        map3d *= auto_sign(ic, threshold=threshold)
        nifti.NiftiImage(map3d.T, header).save(
                                pjoin(output_dir, 'ic_no_thr%02i.nii' % index)
                            )
        
        map3d[np.abs(map3d) < threshold] = 0
        nifti.NiftiImage(map3d.T, header).save(
                                pjoin(output_dir, 'ic%02i.nii' % index)
                            )
        x, y, z = am.find_cut_coords(map3d, mask=mask,
                                        activation_threshold=1e-10)
        # XXX: This is all wrong: we are struggling, jumping from voxel to
        # Talairach.
        y, x, z = am.coord_transform(x, y, z, sform)
        if np.any(map3d != 0):
            # Ugly trick to force the colormap to be symetric:
            map3d[0, 0, 0] = -map3d.max()
            map3d = np.ma.masked_equal(map3d, 0, copy=False)
            if titles is not None:
                title = titles[index]
            else:
                title = ''
            am.plot_map_2d(map3d, sform, (x, y, z), figure_num=512,
                                                    title=title, cmap=cmap)
        else:
            pl.clf()
        pl.savefig(pjoin(output_dir, 'map_%02i.png' % index))


# EOF ##########################################################################
