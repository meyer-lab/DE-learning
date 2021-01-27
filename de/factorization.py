import numpy as np
from scipy.stats import gmean
from scipy.special import expit, logit


alpha = 0.1


def calcW(data, eta, alpha):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    B = (data * alpha) / eta[:, np.newaxis]
    assert np.all(np.isfinite(B))
    B = logit(np.clip(B, 0.05, 0.95))
    assert np.all(np.isfinite(B))
    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data, w, alpha):
    """Directly calculate eta."""
    eta = (alpha * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)


def factorizeEstimate(data, niter=20):
    """ Initialize the parameters based on the data. """
    assert niter > 0
    # TODO: Add tolerance for termination.
    w = np.zeros((data.shape[0], data.shape[0]))

    # Use the data to try and initialize the parameters
    for _ in range(niter):
        eta = calcEta(data, w, alpha)
        assert np.all(np.isfinite(eta))
        assert eta.shape == (data.shape[0], )
        w = calcW(data, eta, alpha)
        assert np.all(np.isfinite(w))
        assert w.shape == (data.shape[0], data.shape[0])

    return w, eta
