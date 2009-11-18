""" Generators used to dispatch cross-validations.
"""
import itertools

def dispatch(n, n_packs):
    """ A generator to dispatch n into n_packs.

        Parameters
        ==========
        n: int
            The number to divide in pack
        n_packs: int
            The number of packs to make
        Returns
        =======
        packs: generator of integers
            A generator of n_pack integers adding up to n

        Examples
        ========
        >>> for n in dispatch(5, 3):
        ...     print n
        2
        2
        1
    """
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            yield this_n


def dispatch_lst(lst, n_packs, total_num=None, return_iterator=False):
    """ A generator to dispatch a list into n_packs.

        Parameters
        ===========
        lst: iterable
            The list or iterable to dispatch
        n_packs: int
            The number of packs to make
        total_num: int, optional
            The total number of elements to draw. If None defaults
            to the length the list. This argument is useful if you which 
            to pass in a generator and have it lazyly loaded.
        return_iterator: boolean, optional
            If True, iterators are returned instead of lists

        Returns
        =======
        packs: generator of sub lists

        Examples
        ========
        >>> lst = range(11)
        >>> for sub_lst in dispatch_lst(lst, 3):
        ...     print sub_lst
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9, 10]

    """
    if total_num is None:
        total_num = len(list(lst))
    iterator = iter(lst)
    for pack_num in dispatch(total_num, n_packs):
        if not return_iterator:
            yield [iterator.next() for _ in range(pack_num)]
        else:
            yield (iterator.next() for _ in range(pack_num))


