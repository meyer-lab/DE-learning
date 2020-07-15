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

def unshape_params(w, alpha, eps):
    '''
    Function receives eps, w, and alpha as separate arrays and returns a 1D parameter array.
    p = [eps, w, alpha] as 1D array

    '''
    p = np.concatenate([eps, w.flatten(), alpha])
    return p

def ODE(t, y, eps, w, alpha):
    '''
    Function receives a set of parameters,
       then return the simulation of ODE over time.
    '''
    return eps * (1 + np.tanh(np.dot(w, y))) - alpha * y

def solver(N_t, p, tp):
    '''
    Function receives time series, a set of parameters and simulation type,
       then return the simulation of ODE overtime/ at last timepoints.
       p = [eps, w, alpha] as 1D array
    '''
    x0 = np.zeros(83) #initial values
    eps, w, alpha = reshape_params(p)
    if tp == 'overtime':
        t = np.arange(N_t)
        sol = np.transpose(solve_ivp(ODE, (0, N_t), x0, args=(eps, w, alpha), t_eval=t, method="LSODA").y)
    elif tp == 'endpoint':
        sol = np.transpose(solve_ivp(ODE, (0, N_t), x0, args=(eps, w, alpha), t_eval=[N_t], method="LSODA").y)
    return sol

def sim_KO(params, geneNum, N_t, tp):
    '''
    Function receives simulation of ODE, 1D array of optimizable parameters, time series, initial values and gene to be knockout.
    Runs the ODE model and returns solution at last timepoint.
    '''
    eps, w, alpha = reshape_params(params)
    w[:, geneNum] = 0.0 #Remove the effect of one gene across all others to simulate the KO experiments
    params = unshape_params(w, alpha, eps)
    sol = solver(N_t, params, tp)
    return sol
