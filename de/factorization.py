import numpy as np
from scipy.stats import gmean
from scipy.optimize import least_squares
from .logistic import logisticF, invlogF


alpha = 0.1


def calcW(data, U, eta, p):
    """Directly calculate w."""
    B = (data * alpha) / eta[:, np.newaxis]
    assert np.all(np.isfinite(B))
    B = invlogF(p, np.clip(B, 0.0001, 0.9999))
    assert np.all(np.isfinite(B))
    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data, U, w, p):
    """Directly calculate eta."""
    eta = (alpha * data) / logisticF(p, w @ U)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)


def calcP(data, U, w, eta):
    """ Directly fit p of the logistic function. """
    outside = alpha * data / eta[:, np.newaxis]
    inside = w @ U

    res = least_squares(lambda x: (logisticF(x, inside) - outside).flatten(), np.ones(3), jac="3-point", bounds = (0.1, 10.0))
    assert res.success
    return res.x


def factorizeEstimate(data, tol=1e-9, maxiter=100):
    """ Initialize the parameters based on the data. """
    assert maxiter > 0
    # Setup initial parameters
    w = np.zeros((data.shape[0], data.shape[0]))
    p = np.ones(3)
    # Make the U matrix
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    # Use the data to try and initialize the parameters
    for ii in range(maxiter):
        eta = calcEta(data, U, w, p)
        assert np.all(np.isfinite(eta))
        assert eta.shape == (data.shape[0], )
        w = calcW(data, U, eta, p)
        assert np.all(np.isfinite(w))
        assert w.shape == (data.shape[0], data.shape[0])
        p = calcP(data, U, w, eta)

        cost = np.linalg.norm(eta[:, np.newaxis] * logisticF(p, w @ U) - alpha * data)

        if ii > 10 and (costLast - cost) < tol:
            # TODO: I believe the cost should be strictly decreasing, so look into this.
            break

        costLast = cost

    return (w, eta, p), cost
