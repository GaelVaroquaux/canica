"""
Analysis pipeline to build a generativ model from fMRI data with no
regressor, eg resting data.
"""

import os

# Major scientific libraries import
import numpy as np
from scipy import stats

import nifti

from joblib import PrintTime
from joblib.make import make, MemMappedNiftiFile, NumpyFile, PickleFile, \
        NiftiFile

# Local imports
from rs.io_context import IOContext, ConfigDict
from rs.read_series import load_session
from rs.tools.ica import fastica

#force_reload = True
force_reload = False 

CONFIG_FILE = 'generativ_model.conf'
#CONFIG_FILE = 'localizer.conf'

this_config = IOContext(
            extra_options=ConfigDict().fromfile(CONFIG_FILE))


################################################################################
# Component extraction on one session/subject
################################################################################

#-------------------------------------------------------------------------------
# PCA: separation of observation noise
#-------------------------------------------------------------------------------

def pca(series):
    """ A PCA routine, used to be able to isolate inputs and outputs
    in joblib's 'make'.
    """
    components, loadings, _ = np.linalg.svd(series, full_matrices=False)
    return components, loadings

#-------------------------------------------------------------------------------
# Per-subject series extraction and PCA. 
#-------------------------------------------------------------------------------

def subject_pca(subject, session=1, n_pca_components=75,
                io_context=None):
    """ Do the preprocessing and calculate the PCA components,.
    """
    #---------------------------------------------------------------------------
    # The IO context: input file patterns, and output paths:
    if io_context is None:
        io_context = ConfigDict().fromfile(CONFIG_FILE)
    # Make sure we do have a rich object, rather than a simple
    # dictionnary.
    for key in io_context.keys():
        if key not in ('output_dir', 'cachedir', 'datadir',
                'local_datadir', 'grey_matter_pattern', 'fmri_data_pattern'):
            io_context.pop(key)
    io_context = IOContext(extra_options=io_context)
    io_context.output_dir = os.path.join(io_context.output_dir, 
                    'subject%i_session%i' % (subject, session))
    io_context.cachedir = os.path.join(io_context.output_dir, 'cache')

    print_time = \
            PrintTime(logfile=io_context.outpath('generativ_model.log'))

    #---------------------------------------------------------------------------
    # Data preprocessing and loading.
    brain_mask, series, header, logvar, mean = \
        make(func=load_session,
            force=force_reload,
            cachedir=io_context.cachedir,
            debug=True,
            output=(MemMappedNiftiFile(io_context.outpath('mask.nii')),
                    NumpyFile(io_context.outpath('series.npy'),
                                                    mmap_mode='r'),
                    PickleFile(io_context.outpath('nifti_header.pkl')),
                    MemMappedNiftiFile(io_context.outpath('logvar.nii')),
                    MemMappedNiftiFile(io_context.outpath('mean.nii')),
                    ),
            )(subject=subject,
            session=session,
            io_context=io_context.copy())

    print_time('Loading time courses for subject %i' % subject)

    #---------------------------------------------------------------------------
    # PCA
    components, loadings = make(func=pca,
                    force=force_reload,
                    cachedir=io_context.cachedir,
                    debug=True,
                    output=[NumpyFile(io_context.outpath('components.npy'),
                                                    mmap_mode='r'),
                            NumpyFile(io_context.outpath('loadings.npy'),
                                                    mmap_mode='r'),
                            ],
                )(series)
    
    print_time('Computing principal components for subject %i' % subject)

#-------------------------------------------------------------------------------
# Load subject components with same mask
#-------------------------------------------------------------------------------

def load_common_mask(dirname, subjects, session):
    """ Find a common mask between the different subjects.
    """
    mask = None
    for subject in subjects:
        mask_file = os.path.join(dirname,
                    'subject%i_session%i' % (subject, session),
                    'mask.nii')
        if mask is None:
            mask = nifti.NiftiImage(mask_file).asarray().T
        else:
            mask += nifti.NiftiImage(mask_file).asarray().T

    mask = ( mask > np.median(mask) )
    return mask


def load_subjects(mask, subjects, dirname, session, n_pca_components):
    """ Load all the PCA components for the different subjects.
    """
    components = list()
    loadings = list()
    for subject in subjects:
        component_file = os.path.join(dirname,
                    'subject%i_session%i' % (subject, session),
                    'components.npy')
        component = np.load(component_file, mmap_mode='r')
        loadings_file = os.path.join(dirname,
                    'subject%i_session%i' % (subject, session),
                    'loadings.npy')
        loadings.append(np.load(loadings_file,
                                    mmap_mode='r')[:n_pca_components])
        mask_file = os.path.join(dirname,
                    'subject%i_session%i' % (subject, session),
                    'mask.nii')
        this_mask = nifti.NiftiImage(mask_file).asarray().T
        this_map = np.zeros(list(this_mask.shape) + [n_pca_components])
        this_map[this_mask.astype(np.bool), :] = \
                                        component[:, :n_pca_components]
        component = this_map[mask, :]
        component /= np.sqrt((component**2).sum(axis=0))
        components.append(component)

    loadings = np.array(loadings)
    return np.hstack(components), [component.shape[-1] for component in
                                    components], loadings


#-------------------------------------------------------------------------------
# Main entry point do dispatch intra-subject processing 
#-------------------------------------------------------------------------------

def intra_subject_pcas(n_jobs=1):
    """ Calculate principal components over different subjects and load
        them in a common mask.
    """
    # Do the intra-subject PCAs
    if n_jobs > 1:
        from multiprocessing import Pool
        pool = Pool(n_jobs)
        apply = pool.apply_async
    else:
        from __builtin__ import apply
    for subject in this_config.subjects:
        apply(subject_pca, 
                (subject, 
                    this_config.session, 
                    this_config.n_pca_components,
                    this_config.copy(),
                ))
    if n_jobs > 1:
        pool.close()
        pool.join()

    fmri_header = PickleFile(
                    this_config.outpath('subject1_session1/nifti_header.pkl')
                    ).load()

    mask = make(func=load_common_mask,
                force=force_reload,
                debug=True,
                cachedir=this_config.cachedir,
                output=NiftiFile(this_config.outpath('common_mask.nii'),
                                 dtype=np.bool, header=fmri_header),
            )(this_config.output_dir, this_config.subjects,
                this_config.session)

    pcas, pca_dims, pca_loadings = make(func=load_subjects,
                          force=force_reload,
                          debug=True,
                          cachedir=this_config.cachedir,
                          output=(
                            NumpyFile(this_config.outpath('pcas.npy'),
                                  mmap_mode='r'),
                            PickleFile(this_config.outpath('ca_dims.pkl')),
                            NumpyFile(this_config.outpath('pca_loadings.npy'),
                                  mmap_mode='r'),
                            ),
                        )(mask, this_config.subjects, 
                               this_config.output_dir,
                               this_config.session,
                               this_config.n_pca_components)
    return pcas, fmri_header, mask, pca_loadings



################################################################################
# Inter-subject extraction of ICA maps
################################################################################

def ica_after_cca(pcas, ccs_threshold=1.6):
    cca_maps, ccs, V = np.linalg.svd(pcas, full_matrices=False)
    n_cca_components = np.argmin(ccs > ccs_threshold)
    _, common_icas = fastica(cca_maps[:, :n_cca_components], 
                                n_cca_components, whiten=False)
    common_icas = common_icas.T
    # Normalize to 1 the ICA maps:
    normed_icas = common_icas/float(cca_maps.shape[0])

    # Project the ICAs on the CCA maps to give a 'cross-subject
    # reproducibility' score.
    proj = np.dot(normed_icas, cca_maps[:, :n_cca_components])
    reproducibility_score = (np.abs(proj)*ccs[:n_cca_components]).sum(axis=-1)

    order = np.argsort(reproducibility_score)[::-1]

    common_icas = common_icas[order, :]

    return common_icas



################################################################################
# Actual computation of the generativ model
################################################################################
if __name__ == '__main__':
    print_time = \
                PrintTime(logfile=this_config.outpath('generativ_model.log'))

    #-------------------------------------------------------------------------
    # Load the intra-subject components
    #-------------------------------------------------------------------------
    pcas, fmri_header, mask, _ = intra_subject_pcas(
                                n_jobs=this_config.n_proc_intrasubject)

    #-------------------------------------------------------------------------
    # Inter-subject CCA and ICA 
    #-------------------------------------------------------------------------
    common_icas = make(func=ica_after_cca,
                       force=force_reload,
                       debug=True,
                       cachedir=this_config.cachedir,
                       output=NumpyFile(
                                this_config.outpath('common_icas.npy'),
                                                    mmap_mode='r'),
                       )(pcas, ccs_threshold=1.65)

    print_time('Computing common ICA decomposition')

    #-------------------------------------------------------------------------
    # Save output maps to nifti
    #-------------------------------------------------------------------------
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

    outdir = this_config.outpath('canica_maps')
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
