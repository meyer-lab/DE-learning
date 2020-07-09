'''
This file contains ODE equations, solver and calculator for Jacobian Matrix
'''
import numpy as np
from scipy.integrate import solve_ivp
import autograd.numpy as anp
from autograd import jacobian
from numba import njit

def solver(n_x, N, x0, p, beta, N_t):
    '''
    Function receives time series and a set of parameters,
       then return the simulation of ODE.
       p = [eps, w, alpha] as 1D array
    '''
    eps = p[0:n_x]
    w = p[n_x:(n_x ** 2 + n_x)].reshape((n_x, n_x))
    alpha = p[(n_x ** 2 + n_x):]
    sol = np.ones((N, N_t, n_x))
    t = np.arange(N_t)
    for i in range(N):
        sol[i, :, :] = np.transpose(solve_ivp(ODE, (0, N_t), x0, args=(eps, w, alpha, beta[:, i]), t_eval=t, method="LSODA").y)
    return t, sol

@njit
def ODE(t, y, eps, w, alpha, beta):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
                 beta: Knock-out effects
    '''
    return eps * (1 + np.tanh(np.dot(w, y))) - (alpha*beta) * y

def jacobian_autograd(t, y, eps, w, alpha, beta):
    '''
    Given a set of parameters and the state of system, it will return the Jacobian of the system.
    '''
    return jacobian(ODE_anp)(y, eps, w, alpha, beta)

def ODE_anp(y, eps, w, alpha, beta):
    '''
    Autograd-packed ODE function.
    '''
    return eps * (1 + anp.tanh(anp.dot(w, y))) - (alpha*beta) * y
