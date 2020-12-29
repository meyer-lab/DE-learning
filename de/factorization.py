import numpy as np
from scipy.linalg import pinv


alpha = 0.1


def calcW(data, eta, alpha):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    Uinv = pinv(U.T)

    B = data * alpha / eta - 1
    B = np.arctanh(np.clip(B, -0.9, 0.9))

    w = (Uinv @ B.T).T
    return w


def calcEta(data, w, alpha):
    """Directly calculate eta."""
    A = np.clip(alpha * data, 0.1, np.inf)
    B = np.clip(1 + np.tanh(w @ data), 0.1, np.inf)
    m = np.exp(np.mean(np.log(A / B), axis=1))
    return np.squeeze(m)


def factorizeEstimate(data):
    """ Initialize the parameters based on the data. """
    # TODO: Add tolerance for termination.
    w = np.zeros(data.shape)

    # Use the data to try and initialize the parameters
    for _ in range(20):
        eta = calcEta(data, w, alpha)
        w = calcW(data, eta, alpha)

    return w, eta
