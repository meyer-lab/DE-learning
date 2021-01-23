'''
Test the factorization model.
'''
import pytest
import numpy as np
from ..factorization import factorizeEstimate, alpha
from ..fitting import runOptim


@pytest.mark.parametrize("sizze", [(8, 8), (12, 13), (15, 14)])
def test_factorizeEstimate(sizze):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.lognormal(size=sizze)
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    w, eta = factorizeEstimate(data, niter=2)
    assert w.shape == (sizze[0], sizze[0])
    assert eta.shape == (sizze[0], )

    wLess, etaLess = factorizeEstimate(data, niter=1)
    costOne = np.linalg.norm(eta[:, np.newaxis] * (1 + np.tanh(w @ U)) - alpha * data)
    costTwo = np.linalg.norm(etaLess[:, np.newaxis] * (1 + np.tanh(wLess @ U)) - alpha * data)
    assert costOne < costTwo


@pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
def test_factorizeBlank(level):
    """ Test that if gene expression is flat we get a blank w. """
    data = np.ones((12, 12)) * level
    w, eta = factorizeEstimate(data)

    np.testing.assert_allclose(w, 0.0, atol=1e-9)
    np.testing.assert_allclose(eta, level * alpha)


@pytest.mark.parametrize("sizze", [(8, 8), (12, 13), (15, 14)])
def test_fit(sizze):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.lognormal(size=sizze)
    outt = runOptim(data, niter=2, disp=False)
    assert np.all(np.isfinite(outt))
