""" Contains ODE equation solvers. """

import numpy as np
from scipy.special import expit
from scipy.integrate import odeint


def solver(ps, ts):
    """ Receives time series and a set of parameters, then return the simulation of ODE. """
    w = np.reshape(ps[:6889], (83, 83))
    alpha = ps[6889:6972]
    eps = ps[6972:]

    assert alpha.size == w.shape[0]
    assert eps.size == w.shape[0]
    assert w.shape[1] == w.shape[0]

    x0 = eps / alpha
    sol = odeint(ODE, x0, ts, args=(eps, w, alpha))
    return sol


def ODE(y, _, eps, w, alpha):
    """ Returns ODE equation. 

    :param eps: Bounds the saturation effect
    :type eps: Any
    :param w: A matrix showing interaction between components
    :type w: ndarray
    :param alpha: Degradation rate
    :type alpha: Any
    """
    return eps * expit(np.dot(w, y)) - alpha * y
