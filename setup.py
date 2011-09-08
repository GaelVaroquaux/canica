
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
def find_packages(path='.'):
    out_list = list()
    for file_name in os.listdir(path):
        file_name = os.path.join(path, file_name)
        if not os.path.isdir(file_name):
            continue
        if os.path.exists(os.path.join(file_name, '__init__.py')):
            package_path = '.'.join(file_name.split('/')[1:])
            out_list.append(package_path)
            out_list.extend(find_packages(path=file_name))
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
