"""
A globber with pattern replacement.
"""

# Standard library imports
import os
import glob

class SuperGlober(object):
    """ A callable object that lists filenames according to a pattern
        with string substitutions.

        In the given path, '~' are expanded to the user home directory.
    """

    def __init__(self, pattern):
        self.pattern = os.path.expanduser(pattern)

    def __call__(self, **substitutions):
        filenames = self._expand_substitutions(substitutions)

        return filenames

    def _expand_substitutions(self, substitutions):
        """ Recursively expand iterables in the substitutions, and call
            the glob on the full expanded dictionnary.
        """
        for key, value in substitutions.iteritems():
            if hasattr(value, '__iter__'):
                output = list()
                for this_value in value:
                    these_substitutions = substitutions.copy()
                    these_substitutions[key] = this_value
                    output.append(self._expand_substitutions(
                                            these_substitutions))
                return output
        else:
            return self._do_glob(substitutions)

    def _do_glob(self, substitutions):
        """ Do the actual glob for a given set of substitutions.
        """
        glob_path = self.pattern % substitutions
        relative_path = os.path.dirname(glob_path)
        these_filenames = glob.glob(glob_path)
        these_filenames.sort()

        return these_filenames


def super_glob(pattern, **substitutions):
    """ List filenames according to a pattern with substitutions.

        Parameters
        ----------
        pattern: string
            The file path, with glob wildcards (such as '*.py') and named
            Python string substitutions (such as '%%(extension)s').
        substitutions: 
            Keywords named with the substitutions' names. Substitutions
            can be given as a list of substitutions.

        Results
        --------
        matches: list
            For each combination of the substitutions, a list of matches.
            Unlike with glob.glob, results are sorted alphabetically.

        Examples
        ---------

        ::
            In [2]: super_glob('tests/testdir/')
            Out[2]: ['tests/testdir/']

            In [3]: super_glob('tests/testdir/*.a')
            Out[3]: 
            ['tests/testdir/ab.a',
             'tests/testdir/ac.a']

            In [4]: super_glob('tests/testdir/*.%(ext)s', ext='a')
            Out[4]: 
            ['tests/testdir/ab.a',
             'tests/testdir/ac.a']

            In [5]: super_glob('tests/testdir/*.%(ext)s', ext=['a', 'b'])
            Out[5]: 
            [['tests/testdir/ab.a',
              'tests/testdir/ac.a'],
             ['tests/testdir/ab.b', 
              'tests/testdir/ac.b']]

    """
    super_globber = SuperGlober(pattern)
    return super_globber(**substitutions)

