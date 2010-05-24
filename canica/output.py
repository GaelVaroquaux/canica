"""
Visualization functions for CanICA.
"""
import os
from os.path import join as pjoin

# Major scientific library imports
import numpy as np

# Neuroimaging library imports
from nipy.io.imageformats import save, Nifti1Image, Nifti1Header


def auto_sign(map, threshold=0):
    positiv_mass = (map > threshold).sum()
    negativ_mass = (map < -threshold).sum()
    if negativ_mass > positiv_mass:
        return -1
    return 1


def save_ics(icas, mask, threshold, output_dir, header, mean=None):
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

    header['cal_max'] = np.nanmax(icas)
    header['cal_min'] = np.nanmin(icas)

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

    # Save the mask
    mask_header = header.copy()
    mask_header['cal_min'] = 0
    mask_header['cal_max'] = 1
    save(Nifti1Image(mask, affine, header=mask_header),
                        pjoin(output_dir, 'mask.nii')
                        )

    if mean is not None:
        # save the mean
        mean_img = np.zeros(maps3d.shape[:-1])
        mean_img[mask] = mean
        mean_header = header.copy()
        mean_header['cal_min'] = mean.min()
        mean_header['cal_max'] = mean.max()
        save(Nifti1Image(mean_img, affine, header=mean_header),
                            pjoin(output_dir, 'mean.nii')
                            )
        return maps3d, affine, mean_img

    return maps3d, affine


# EOF ##########################################################################
