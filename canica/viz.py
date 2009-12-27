"""
Visualization functions for CanICA.
"""
import os
from os.path import join as pjoin

# Major scientific library imports
import numpy as np
import pylab as pl

# Neuroimaging library imports
from nipy.io.imageformats import save, Nifti1Image, Nifti1Header
from nipy.neurospin.viz import activation_maps as am
from nipy.neurospin.datasets import VolumeImg


def auto_sign(map, threshold=0):
    positiv_mass = (map > threshold).sum()
    negativ_mass = (map < -threshold).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1


def save_ics(icas, mask, threshold, output_dir, header, titles=None,
                format=None, cmap=am.cm.cold_hot, mean_img=None,
                **kwargs):
    """ Save the independant compnents to Nifti and to images.

        Parameters
        -----------
        icas: 2D ndarray
            The independant components, as returned by CanICA.
        mask: 3D ndarray
            The 3D boolean array used to go from index to voxel space for
            each IC.
        threshold: float
            The threshold value used to threshold the ICs
        output_dir: string
            The directory in which maps and nifti files will be saved.
        header: dictionnary or Nifti1Header object
            The header for the nifti file, as returned by pynifti's
            header attribute, or
            nipy.io.imageformats.load().get_header().
        titles: None or list of strings.
            Titles to be used for each ICs
	format: {None, 'png', 'pdf', 'svg', 'jpg', ...}
	    The format used to save a preview. If None, no preview is saved.
        cmap: matplotlib colormap
            The colormap to be used for the independant compnents, for 
            example pylab.cm.hot
    """
    # put in order n_ic, n_voxels
    icas = icas.T
    mask = mask.astype(np.bool)

    if isinstance(header, Nifti1Header):
        sform = header.get_best_affine()
    else:
        sform = header['sform']
        # XXX: I don't know how to convert a dictionnary to a
        # Nifti1Header.
        header = None

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if mean_img is not None:
        anat = np.zeros(mask.shape)
        anat[mask] = mean_img
        anat = np.ma.masked_array(anat, np.logical_not(mask))
        img = VolumeImg(anat, affine=sform, world_space='mine')
        img = img.xyz_ordered()
        kwargs['anat'] = img.get_data()
        kwargs['anat_affine'] = img.affine
        
    for index, ic in enumerate(icas):
        print 'Outputing map %i out of %i' % (index + 1, len(icas)) 
        map3d = np.zeros(mask.shape)
        map3d[mask] = ic
        # Modify the 3D map rather than the IC to avoid modifying
        # the original data.
        map3d *= auto_sign(ic, threshold=threshold)
        save(Nifti1Image(map3d, sform, header=header),
                          pjoin(output_dir, 'ic_no_thr%02i.nii' % index)
                        )
        
        map3d[np.abs(map3d) < threshold] = 0
        save(Nifti1Image(map3d, sform, header=header),
                            pjoin(output_dir, 'ic%02i.nii' % index)
                        )
        if format is None:
            continue
        img = VolumeImg(map3d, affine=sform, world_space='mine')
        img = img.xyz_ordered()
        map3d = img.get_data()
        this_affine = img.affine
        img = VolumeImg(mask, affine=sform, world_space='mine',
                              interpolation='nearest')
        img = img.xyz_ordered()
        mask_xyz = img.get_data()

        x, y, z = am.find_cut_coords(map3d, mask=mask_xyz,
                                        activation_threshold=1e-10)
        x, y, z = am.coord_transform(x, y, z, this_affine)
        pl.clf()
        if np.any(map3d != 0):
            # Force the colormap to be symetric:
            map_max = max(-map3d.min(), map3d.max())
            kwargs['vmin'] = -map_max
            kwargs['vmax'] =  map_max
            map3d = np.ma.masked_equal(map3d, 0, copy=False)
            if titles is not None:
                title = titles[index]
            else:
                title = None
            am.plot_map_2d(map3d, this_affine, (x, y, z), figure=512,
                                                    title=title,
                                                    cmap=cmap,
                                                    **kwargs)
        pl.savefig(pjoin(output_dir, 'map_%02i.%s' % (index, format)))


# EOF ##########################################################################
