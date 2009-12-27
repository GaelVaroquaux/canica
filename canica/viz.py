"""
Visualization functions for CanICA.
"""
import os
from os.path import join as pjoin
import time

# Major scientific library imports
import numpy as np
import pylab as pl

# Neuroimaging library imports
from nipy.io.imageformats import save, Nifti1Image, Nifti1Header
from nipy.neurospin.viz import activation_maps as am
from nipy.neurospin.datasets import VolumeImg

from .tools import markup

def auto_sign(map, threshold=0):
    positiv_mass = (map > threshold).sum()
    negativ_mass = (map < -threshold).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1


def save_ics(icas, mask, threshold, output_dir, header):
    """ Save the independant compnents to Nifti.

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
    """
    # put in order n_ic, n_voxels
    icas = icas.T
    mask = mask.astype(np.bool)

    if isinstance(header, Nifti1Header):
        affine = header.get_best_affine()
    else:
        affine = header['sform']
        # XXX: I don't know how to convert a dictionnary to a
        # Nifti1Header.
        header = None

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    icas = icas.copy()
    for ic in icas:
        ic *= auto_sign(ic, threshold=threshold)

    maps3d = np.zeros(list(mask.shape) + [len(icas)])

    maps3d[mask] = icas.T
    save(Nifti1Image(maps3d, affine, header=header),
                          pjoin(output_dir, 'icas_no_thr.nii')
                        )
    maps3d[np.abs(maps3d) < threshold] = 0
    save(Nifti1Image(maps3d, affine, header=header),
                          pjoin(output_dir, 'icas.nii')
                        )
    return maps3d, affine


def plot_ics(maps3d, affine, output_dir, titles=None, 
            format='png', cmap=am.cm.cold_hot, mean_img=None,
            report=True, **kwargs):
    """ Save the ics to image file, and to a report.

        Parameters
        ----------
        titles: None, string or list of strings.
            Titles to be used for each ICs
        format: {'png', 'pdf', 'svg', 'jpg', ...}
            The format used to save a preview.
        cmap: matplotlib colormap
            The colormap to be used for the independant compnents, for 
            example pylab.cm.hot
        report: boolean, optional
            If report is True, an html report is saved.
        kwargs:
            Extra keyword arguments are passed to plot_map_2d.
    """
    if mean_img is not None:
        img = VolumeImg(mean_img, affine=affine, world_space='mine')
        img = img.xyz_ordered()
        kwargs['anat'] = img.get_data()
        kwargs['anat_affine'] = img.affine

    img_files = list()
    maps3d = np.rollaxis(maps3d, -1)
    for index, map3d in enumerate(maps3d):
        print 'Outputing image for map %i out of %i' \
            % (index, len(maps3d))
        # XYZ order the images, this should be done in the viz
        # code.
        img = VolumeImg(map3d, affine=affine, world_space='mine')
        img = img.xyz_ordered()
        map3d = img.get_data()
        this_affine = img.affine
        x, y, z = am.find_cut_coords(map3d, activation_threshold=1e-10)
        x, y, z = am.coord_transform(x, y, z, this_affine)
        if np.any(map3d != 0):
            # Force the colormap to be symetric:
            map_max = max(-map3d.min(), map3d.max())
            kwargs['vmin'] = -map_max
            kwargs['vmax'] =  map_max
            map3d = np.ma.masked_equal(map3d, 0, copy=False)
            if titles is not None:
                if isinstance(titles, basestring):
                    title = titles % locals()
                else:
                    title = titles[index]
            else:
                title = None
            am.plot_map_2d(map3d, this_affine, (x, y, z), figure=512,
                                                    title=title,
                                                    cmap=cmap,
                                                    **kwargs)
        img_file = 'map_%02i.%s' % (index, format)
        pl.savefig(pjoin(output_dir, img_file))
        pl.clf()
        img_files.append(img_file)

    if format in ('png', 'jpg'):
        report = markup.page()
        report.init(title='CanICA report')

        report.p(""" CanICA run, %s.""" % time.asctime())

        report.img(src=img_files)
        file(pjoin(output_dir, 'canica_report.html'), 'w').write(str(report))


# EOF ##########################################################################
