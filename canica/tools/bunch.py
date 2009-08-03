"""
The bunch pattern.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


class Bunch(dict):
    """ A dict that exposes its keys as attributes.
    """

    def __init__(self, *args, **kwargs):
        """ >>> c = Bunch(a=1)
            >>> 'a' in c
            True
            >>> c.a == c['a']
            True
        """
        self.__dict__ = self
        dict.__init__(self, *args, **kwargs)

