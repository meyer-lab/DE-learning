'''
Test the factorization model.
'''
import pytest
import numpy as np
from scipy.special import expit
from ..factorization import factorizeEstimate, alpha
from ..fitting import runOptim
from ..importData import formMatrix


def test_factorizeEstimate():
    """ Test that this runs successfully with reasonable input. """
    data = formMatrix()
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    w, eta = factorizeEstimate(data)
    assert w.shape == (data.shape[0], data.shape[0])
    assert eta.shape == (data.shape[0], )

    wLess, etaLess = factorizeEstimate(data, maxiter=1)
    costOne = np.linalg.norm(eta[:, np.newaxis] * expit(w @ U) - alpha * data)
    costTwo = np.linalg.norm(etaLess[:, np.newaxis] * expit(wLess @ U) - alpha * data)
    assert costOne < costTwo


@pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
def test_factorizeBlank(level):
    """ Test that if gene expression is flat we get a blank w. """
    data = np.ones((12, 12)) * level
    w, eta = factorizeEstimate(data, maxiter=2)

    np.testing.assert_allclose(w, 0.0, atol=1e-9)
    np.testing.assert_allclose(eta, 2 * level * alpha)


@pytest.mark.parametrize("sizze", [(8, 8), (12, 13), (15, 14)])
def test_fit(sizze):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.lognormal(size=sizze)
    outt = runOptim(data, niter=20, disp=False)
    assert np.all(np.isfinite(outt))
