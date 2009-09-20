"""
Tests for the super glob.
"""
import os
from os.path import join as pjoin

import nose

from ..super_glob import super_glob

dirname = pjoin(os.path.dirname(__file__), 'testdir')

def test_super_glob_dir():
    """ Test the super_glob on a directory.
    """
    nose.tools.assert_equal(
                            [dirname],
                            super_glob(dirname)
                            )


def test_super_glob_files():
    """ Test the super_glob on a simple file pattern.
    """
    nose.tools.assert_equal(
                            [pjoin(dirname, f) for f in 'ab.a', 'ac.a'],
                            super_glob(pjoin(dirname, '*.a'))
                            )


def test_super_glob_subs():
    """ Test the super_glob on a simple subsitution.
    """
    nose.tools.assert_equal(
                            [pjoin(dirname, f) for f in 'ab.a', 'ac.a'],
                            super_glob(pjoin(dirname, '*.%(ext)s'),
                                       ext='a')
                            )

def test_super_glob_multi_subs():
    """ Test the super_glob on multiple list subsitution.
    """
    nose.tools.assert_equal(
                            [[pjoin(dirname, f) for f 
                                    in 'ab.%s' % e, 'ac.%s' % e]
                                    for e in 'a', 'b'],
                            super_glob(pjoin(dirname, '*.%(ext)s'),
                                       ext=['a', 'b'])
                            )


