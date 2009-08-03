""" An generator used to dispatch bootstrap numbers.
"""

def dispatch(n, n_packs):
    """ A generator to dispatch n into n_packs.
    """
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            yield this_n


################################################################################
# Tests

def test_dispatch():
    import random
    import nose
    for _ in range(10):
        n = random.randint(50, 100)
        n_packs = random.randint(1, 10)
        n_ = 0
        for this_n in dispatch(n, n_packs):
            n_ += this_n
        yield nose.tools.assert_equal, n_, n
        

