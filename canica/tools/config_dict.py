"""
ConfigDict class: 

Dictionnary-like object that exposes its keys as attributes, and can be 
persisted to file.
"""

import re
import warnings

class ConfigDict(dict):
    """ Dictionnary-like object that exposes its keys as attributes, and
        can be persisted to file.
    """

    def __init__(self, **kwargs):
        """ >>> c = ConfigDict(a=1)
            >>> 'a' in c
            True
            >>> c.a == c['a']
            True
        """
        dict.__init__(self, **kwargs)
        self.__dict__ = self

    def tofile(self, filename):
        """ Save the configuration to file.

            The format of the file is:
                key = value
        """
        outfile = file(filename, 'w')
        for key, value in self.iteritems():
            outfile.write('%s = %s\n' % (key, repr(value)))

    def fromfile(self, filename, override=True):
        """ Read the configuration from file.

            The format of the file is:
                key = value

            If override is False, existing keys in the dictionnary are
            not overriden when loading.
        """
        infile = file(filename, 'r')
        for line_num, line in enumerate(infile):
            match_object = re.match('\s*([a-zA-Z]\w*)\s*=\s*(.*)\s*', line)
            if match_object:
                key, value = match_object.groups()
                if key in self and not override:
                    continue
                self[key] = eval(value)
            elif not re.match('(\s*)|(#.*)', line):
                warnings.warn('Invalid line %i: %s' (line_num, line),  
                                                            stacklevel=2)
        # Convenience function.
        return self


def test_config_dict():
    b = ConfigDict(a=1)
    b.c = 'c'
    assert b.keys() == ['a', 'c', ]
    b.d = 1.
    b.e = True
    import tempfile
    tmpfile = tempfile.mktemp()
    b.tofile(tmpfile)
    c = ConfigDict().fromfile(tmpfile)
    assert c.items() == b.items()

if __name__ == '__main__':
    test_config_dict()
