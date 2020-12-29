'''
Test the factorization model.
'''
import pytest
import numpy as np
from ..factorization import factorizeEstimate, alpha


@pytest.mark.parametrize("sizze", [8, 12, 13])
def test_factorizeEstimate(sizze):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.normal(size=(sizze, sizze))
    w, eta = factorizeEstimate(data)

    assert w.shape == (sizze, sizze)
    assert eta.shape == (sizze, )


@pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
def test_factorizeBlank(level):
    """ Test that if gene expression is flat we get a blank w. """
    data = np.ones((12, 12)) * level
    w, eta = factorizeEstimate(data)

    np.testing.assert_allclose(w, 0.0, atol=1e-9)
    np.testing.assert_allclose(eta, level * alpha)
