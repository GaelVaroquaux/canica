"""
Test the fastica algorithm.
"""

import numpy as np

# No relative import to be able to run this file.
#from ..fastica import fastica, center_and_norm
from canica.algorithms.fastica import fastica, center_and_norm

def test_fastica(demo=False, figure=None):
    """ Test the FastICA algorithm on very simple data.

        Parameters
        ----------
        demo: boolean, optional
            If demo is True, plot a figure to illustrate the algorithm
        figure: int, option
    """

    n_samples = 1000
    # Generate two sources:
    t  = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi*t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array([[np.cos(phi),  np.sin(phi)], 
                       [np.sin(phi), -np.cos(phi)]])
    m  = np.dot(mixing, s)
    center_and_norm(m)
    k_, mixing_, s_ = fastica(m)

    # Check that the mixing model described in the docstring holds:
    np.testing.assert_almost_equal(s_, np.dot(np.dot(mixing_, k_), m))

    center_and_norm(s_)
    s1_, s2_ = s_
    # Check to see if the sources have been estimated in the wrong order
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))

    # Check that we have estimated the original sources
    np.testing.assert_almost_equal(np.dot(s1_, s1)/n_samples, 1, decimal=3)
    np.testing.assert_almost_equal(np.dot(s2_, s2)/n_samples, 1, decimal=3)


    if demo:
        import pylab as pl
        pl.figure(figure, figsize=(9, 7))
        pl.clf()
        pl.plot(t, m[0] + 3,    'g', label='Signal 1')
        pl.plot(t, m[1] + 3,    'c', label='Signal 2')
        pl.plot(t, s1      ,  'r', label='Source 1')
        pl.plot(t, s1_    , 'k--', label='Estimated source 1')
        pl.plot(t, s2 - 3,  'b', label='Source 2')
        pl.plot(t, s2_ - 3, 'k--', label='Estimated source 2')
        pl.legend()
        pl.title('FastICA demo')


if __name__ == '__main__':
    test_fastica(demo=True)
    import pylab as pl
    pl.show()

