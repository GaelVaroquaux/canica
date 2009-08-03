"""
IO of the resting state data.
"""

# Standard library imports
import os
import glob
import shutil
import logging

# Local imports
from io_context import IOContext

io_logger = logging.Logger('IO', level=logging.INFO)

io_logger.addHandler(logging.StreamHandler())

def lazy_copy_file(filename, target_dir):
    """ Copy a file only if it does not exist.
    """
    if not os.path.exists(os.path.join(target_dir, 
                          os.path.basename(filename))):
        shutil.copy(filename, target_dir)
        io_logger.info('Copying %s to %s' 
                        % (filename, target_dir))
    else:
        io_logger.debug('Skipping %s, as it is already in %s' 
                        % (filename, target_dir))


class FileGlober(object):
    """ Base class for functors that list filename according to a pattern
        and (optionally caches them locally).
    """

    def __init__(self, pattern, io_context=None):
        self.pattern = pattern
        self.io_context = io_context

    def __call__(self, io_context=None, **replacements):
        if io_context is None:
            io_context = self.io_context
        if io_context is None:
            io_context = IOContext()
        self.datadir = io_context['datadir']
        self.local_datadir = io_context.get('local_datadir', None)
        # If the upstream data dir is not accessible and there is a
        # local copy, fallback to the local copy
        if not os.path.exists(self.datadir) and self.local_datadir is not None:
            io_logger.warn('Data directory not accessible, using '
                'local cache')
            self.datadir = self.local_datadir

        filenames = self._expand_replacements(replacements)

        return filenames

    def _expand_replacements(self, replacements):
        """ Recursively expand iterables in the replacements, and call
            the glob on the full expanded dictionnary.
        """
        for key, value in replacements.iteritems():
            if hasattr(value, '__iter__'):
                output = list()
                for this_value in value:
                    these_replacements = replacements.copy()
                    these_replacements[key] = this_value
                    output.append(self._expand_replacements(
                                            these_replacements))
                return output
        else:
            return self._do_glob(replacements)

    def _do_glob(self, replacements):
        """ Do the actual glob and copy for a given set of replacements.
        """
        glob_path = self.pattern % replacements
        relative_path = os.path.dirname(glob_path)
        these_filenames = glob.glob(os.path.join(self.datadir, glob_path))
        these_filenames.sort()

        # Mirror the files locally
        if self.local_datadir is not None \
                and not self.local_datadir == self.datadir:
            target_dir = os.path.join(self.local_datadir, relative_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for index, filename in enumerate(these_filenames):
                lazy_copy_file(filename, target_dir)
                img_filename = filename[:-4]+'.img'
                if os.path.exists(img_filename):
                    lazy_copy_file(img_filename, target_dir)
                target_file = os.path.join(target_dir,
                                            os.path.basename(filename))
                # replace the filename in the list by the local
                # filename.
                these_filenames[index] = target_file

        return these_filenames
        


get_fmri_session_files = FileGlober(
        'subject%(subject)i/functional/fMRI/session%(session)i/sw*.hdr'
    #    '%(subject)s/fMRI/asquisition%(session)i/localizer1/swa*.hdr'
    )

get_fmri_session_files.__doc__ = \
    """ Return the file names for the subject and fMRI session.

        INPUTS:
        - subjects: subject number, or list of subject numbers
        - sessions: session number, or list of session numbers
        - io_context: an optional dictionnary with keys:
            - 'datadir': path to the original data
            - 'local_datadir': optional path to a local copy of the data
              that mirrors the NFS data dir.
        OUTPUT:
        A list of absolute paths to the nifti file, or a list of lists (of
        lists) of paths, depending on the shape of the input arguments.
    """

get_grey_matter_masks = FileGlober(
        'subject%(subject)i/anatomy/wc1*.hdr'
    )

get_grey_matter_masks.__doc__ = \
    """ Return the file names of the grey matter masks for the subject.

        INPUTS:
        - subjects: subject number, or list of subject numbers
        - io_context: an optional dictionnary with keys:
            - 'datadir': path to the original data
            - 'local_datadir': optional path to a local copy of the data
              that mirrors the NFS data dir.
        OUTPUT:
        An absolute path to the nifti file, or a list of paths, depending 
        on the shape of the input arguments.
    """


if __name__ == '__main__':
    print get_fmri_session_files(subject=1, session=1)
    print get_grey_matter_masks(subject=1)

