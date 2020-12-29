import numpy as np
from scipy.stats import gmean
from scipy.linalg import pinv


alpha = 0.1


def calcW(data, eta, alpha):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    Uinv = pinv(U.T)

    B = (data * alpha) / eta[:, np.newaxis] - 1
    B = np.arctanh(np.clip(B, -0.9, 0.9))

    return (Uinv @ B.T).T


def calcEta(data, w, alpha):
    """Directly calculate eta."""
    A = np.clip(alpha * data, 0.01, np.inf)
    B = np.clip(1 + np.tanh(w @ data), 0.01, np.inf)
    Am = gmean(A, axis=1)
    Bm = gmean(B, axis=1)
    return Am / Bm


def factorizeEstimate(data, niter=20):
    """ Initialize the parameters based on the data. """
    # TODO: Add tolerance for termination.
    w = np.zeros((data.shape[0], data.shape[0]))

    # Use the data to try and initialize the parameters
    for _ in range(niter):
        eta = calcEta(data, w, alpha)
        assert eta.shape == (data.shape[0], )
        w = calcW(data, eta, alpha)
        assert w.shape == (data.shape[0], data.shape[0])

    return w, eta
