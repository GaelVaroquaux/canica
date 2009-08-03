"""
Test the ConfigDict class.

"""

from ..config_dict import ConfigDict

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

