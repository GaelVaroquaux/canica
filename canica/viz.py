"""
Visualization functions for CanICA.
"""
from os.path import join as pjoin
import time
import pprint

# Major scientific library imports
import numpy as np
import pylab as pl

# Neuroimaging library imports
from nipy.neurospin import viz
from nipy.neurospin.datasets import VolumeImg

from .tools import markup


def plot_ics(maps3d, affine, output_dir, titles=None, 
            format='png', cmap=viz.cm.cold_hot, mean_img=None,
            report=True, 
            parameters=None,
            **kwargs):
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
        parameters: None or dictionnary, optional
            Extra parameters to put in the report.
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
        img_file = pjoin(output_dir, 'map_%02i.%s' % (index, format))
        print 'Outputing image for map %2i out of % 2i:  %s' \
            % (index+1, len(maps3d), img_file)
        # XYZ order the images, this should be done in the viz
        # code.
        img = VolumeImg(map3d, affine=affine, world_space='mine')
        img = img.xyz_ordered()
        map3d = img.get_data()
        this_affine = img.affine
        x, y, z = viz.find_cut_coords(map3d, activation_threshold=1e-10)
        x, y, z = viz.coord_transform(x, y, z, this_affine)
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
            viz.plot_map(map3d, this_affine, (x, y, z), figure=512,
                                                    title=title,
                                                    cmap=cmap,
                                                    **kwargs)
        pl.savefig(img_file)
        pl.clf()
        img_files.append(img_file)

    if format in ('png', 'jpg'):
        report = markup.page()
        report.init(title='CanICA report')

        report.p(""" CanICA run, %s.""" % time.asctime())
        report.h1("Independent components")

        report.img(src=img_files)
        if parameters is not None:
            report.h1("Parameters")
            for name, value in parameters.iteritems():
                if isinstance(value, list) or isinstance(value, tuple):
                    report.p(r'<strong>%s</strong>: %s(' % 
                                (name, value.__class__.__name__))
                    report.ul()
                    descriptions = list()
                    for item in value:
                        description = pprint.pformat(item)
                        if len(description) > 1000:
                            description = ('%s...<br>&nbsp;&nbsp;&nbsp;...%s' 
                                    % (description[:200], description[-200:]))
                            descriptions.append(description)
                    report.li(descriptions)
                    report.ul.close()
                    report.p(')')
                else:
                    description = pprint.pformat(value)
                    if len(description) > 1500:
                        description = ('%s...<br>&nbsp;...%s' 
                                % (description[:500], description[-500:]))
                    report.p(r'<strong>%s</strong>: %s' % 
                                        (name, description))
        report_file = pjoin(output_dir, 'canica_report.html')
        file(report_file, 'w').write(str(report))
        print 80*'_'
        print 'CanICA: report in %s' % report_file


# EOF ##########################################################################
