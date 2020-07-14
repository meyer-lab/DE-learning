'''
This file contains ODE equations, solver and calculator for Jacobian Matrix
'''
import numpy as np
from scipy.integrate import solve_ivp
from os.path import join, dirname
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Base


path_here = dirname(dirname(__file__))
Base.MainInclude.include(join(path_here, "de/model.jl"))

def julia_solver(ps):
    '''
    Function receives a set of parameters,
       then return the simulation of ODE over time.
       p = [eps, w, alpha] as 1D array
    '''
    return jl.eval('solveODE')(ps)

def julia_sol_matrix(ps):
    '''
    Function receives a set of parameters then returns the simulation of ODE at the last timepoint for all KO models,
        mimicking the experimental data.
        p = [eps, w, alpha] as 1D array
    '''
    return jl.eval('sol_matrix')(ps)

def reshape_params(ps):
    '''
    Function receives a set of parameters then returns eps, w, and alpha as separate arrays
    p = [eps, w, alpha] as 1D array
    '''
    eps = ps[0:83]
    w = ps[83:(83 ** 2 + 83)].reshape((83, 83))
    alpha = ps[(83 ** 2 + 83):]
    return eps, w, alpha

def ODE(t, y, eps, w, alpha):
    '''
    Function receives a set of parameters,
       then return the simulation of ODE over time.
    '''
    return eps * (1 + np.tanh(np.dot(w, y))) - alpha * y

def solver(x0, N_t, p):
    '''
    Function receives initial values, time series, and a set of parameters,
       then return the simulation of ODE.
       p = [eps, w, alpha] as 1D array
    '''
    t = np.arange(N_t)
    eps, w, alpha = reshape_params(p)
    sol = np.transpose(solve_ivp(ODE, (0, N_t), x0, args=(eps, w, alpha), t_eval=t, method="LSODA").y)
    return t, sol
