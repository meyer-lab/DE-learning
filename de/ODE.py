"""
This file contains ODE equation solvers
"""
import numpy as np
from scipy.integrate import odeint


def solver(ps, ts):
    '''Function receives time series and a set of parameters,
       then return the simulation of ODE.
    '''
    w = np.reshape(ps[:6889], (83, 83))
    alpha = ps[6889:6972]
    eps = ps[6972:]

    assert alpha.size == w.shape[0]
    assert eps.size == w.shape[0]
    assert w.shape[1] == w.shape[0]

    x0 = eps / alpha
    sol = odeint(ODE, x0, ts, args=(eps, w, alpha))
    return sol


def ODE(y, t, eps, w, alpha):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
    '''
    return eps * np.tanh(np.dot(w, y)) - alpha * y
