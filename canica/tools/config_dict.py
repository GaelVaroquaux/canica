"""
ConfigDict class: 

Dictionnary-like object that exposes its keys as attributes, and can be 
persisted to file.
"""

import re
import warnings

from .bunch import Bunch

class ConfigDict(Bunch):
    """ Dictionnary-like object that exposes its keys as attributes, and
        can be persisted to file.
    """

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


