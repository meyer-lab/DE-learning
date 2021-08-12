""" Contains functions for model parameters. """

import numpy as np
from scipy.stats import gmean
from scipy.special import expit, logit


def calcW(data, eta, alphaIn):
    """ Directly calculate w. """
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    B = (data * alphaIn) / eta[:, np.newaxis]
    assert np.all(np.isfinite(B))
    B = logit(np.clip(B, 0.0001, 0.9999))
    assert np.all(np.isfinite(B))
    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data, w, alphaIn):
    """ Directly calculate eta. """
    eta = (alphaIn * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)
