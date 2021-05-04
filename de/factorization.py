""" Methods to implement the factorization/fitting process. """

import numpy as np
from scipy.stats import gmean
from scipy.special import expit, logit
from .importData import importLINCS


alpha = 0.1


def calcW(data, eta, alphaIn):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    B = (data * alphaIn) / eta[:, np.newaxis]
    assert np.all(np.isfinite(B))
    B = logit(np.clip(B, 0.0001, 0.9999))
    assert np.all(np.isfinite(B))
    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data, w, alphaIn):
    """Directly calculate eta."""
    eta = (alphaIn * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)


def factorizeEstimate(data, tol=1e-9, maxiter=10000):
    """ Initialize the parameters based on the data. """
    assert maxiter > 0
    # TODO: Add tolerance for termination.
    w = np.zeros((data.shape[0], data.shape[0]))
    # Make the U matrix
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    # Use the data to try and initialize the parameters
    for ii in range(maxiter):
        eta = calcEta(data, w, alpha)
        assert np.all(np.isfinite(eta))
        assert eta.shape == (data.shape[0], )
        w = calcW(data, eta, alpha)
        assert np.all(np.isfinite(w))

        assert w.shape == (data.shape[0], data.shape[0])

        cost = np.linalg.norm(eta[:, np.newaxis] * expit(w @ U) - alpha * data)

        if ii > 10 and (costLast - cost) < tol:
            # TODO: I believe the cost should be strictly decreasing, so look into this.
            break

        costLast = cost

    return w, eta


def cellLineFactorization(cellLine):
    data, annotation = importLINCS(cellLine)
    w, eta = factorizeEstimate(data)

    return w, eta, annotation
