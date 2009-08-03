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


