
import sys
import os
from distutils.core import setup

################################################################################
# For some commands, use setuptools

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

################################################################################
# Automatic package discovery, cuz I am tired of forgetting to add them.
def find_packages():
    out_list = list()
    for root, dirs, files in os.walk('.'):
        if '__init__.py' in files:
            package_path = '.'.join(root.split('/')[1:])
            out_list.append(package_path)
    return out_list


################################################################################
# The main setup function
def main(**extra_args):
    setup( name = 'canica',
           description = 'ICA analysis of fMRI data',
           author = 'Gael Varoquaux',
           author_email = 'gael.varoquaux@normalesup.org',
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__,
           packages=find_packages(),
           **extra_args)


if __name__ == "__main__":
    main(**extra_setuptools_args)
