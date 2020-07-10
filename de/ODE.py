'''
This file contains ODE equations, solver and calculator for Jacobian Matrix
'''
import numpy as np
from scipy.integrate import odeint
import autograd.numpy as anp
from autograd import jacobian

def solver(n_x, N, x0, p, beta, dt, N_t):
    '''
    Function receives time series and a set of parameters,
       then return the simulation of ODE.
       p = [eps, w, alpha] as 1D array
    '''
    eps = p[0:n_x]
    w = p[n_x:(n_x ** 2 + n_x)].reshape((n_x,n_x))
    alpha = p[(n_x ** 2 + n_x):]
    t = np.linspace(0, dt*N_t, N_t)
    sol = np.ones((N, N_t, n_x))
    for i in range(N):
        sol[i, :, :] = odeint(ODE, x0, t, args=(eps, w, alpha, beta[:, i]))
    return t, sol

def ODE(y, t, eps, w, alpha, beta):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
                 beta: Knock-out effects
    '''
    return eps * (1 + np.tanh(np.dot(w, y))) - (alpha*beta) * y
  
def jacobian_autograd(y, p, beta, N, n_x):
    '''
    Given a set of parameters and the state of system, it will return the Jacobian of the system.
    '''
    eps = p[0:n_x]
    w = p[n_x:(n_x ** 2 + n_x)].reshape((n_x,n_x))
    alpha = p[(n_x ** 2 + n_x):]
    jac = np.zeros((N, n_x, n_x))
    for i in range(N):
        jac[i, :, :] = jacobian(ODE_anp)(y[i, -1, :], eps, w, alpha, beta[:, i])
    return jac

def ODE_anp(y, eps, w, alpha, beta):
    '''
    Autograd-packed ODE function.
    '''
    return eps * (1 + anp.tanh(anp.dot(w, y))) - (alpha*beta) * y