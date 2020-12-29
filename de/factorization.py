import numpy as np
from scipy.linalg import pinv


def calcW(data, eta, alpha):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    Uinv = sp.linalg.pinv(U.T)

    B = data * alpha / eta - 1
    B = np.arctanh(np.clip(B, -0.9, 0.9))

    w = (Uinv @ B.T).T
    assert w.shape == (83, 83)
    return w


def calcEta(data, w, alpha):
    """Directly calculate eta."""
    A = clamp.(alpha .* data, 0.1, Inf)
    B = clamp.(1 .+ tanh.(w * data), 0.1, Inf)
    m = exp.(mean(log.(A . / B), dims=2))
    return np.squeeze(m)


def factorizeEstimate(data):
    """ Initialize the parameters based on the data. """
    # TODO: Add tolerance for termination.
    w = zeros(data.shape[0], data.shape[0])

    # Use the data to try and initialize the parameters
    for ii in range(20):
        eta = calcEta(data, w, alpha)
        w = calcW(data, eta, alpha)

    return w, eta
