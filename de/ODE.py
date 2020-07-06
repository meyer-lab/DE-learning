'''
This file contains ODE equations and solver
'''
import numpy as np
from scipy.integrate import odeint

def solver(x0, eps, w, alpha, beta, dt, N_t):
    '''Function receives time series and a set of parameters,
       then return the simulation of ODE.
    '''
    t = np.linspace(0, dt*N_t, N_t)
    sol = odeint(ODE, x0, t, args=(eps, w, alpha, beta))
    return t, sol
def ODE(y, t, eps, w, alpha, beta):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
                 beta: Knock-out effects
    '''
    return eps * np.tanh(np.dot(w, y)) - (alpha*beta) * y
