'''
This file contains ODE equation solvers from Julia
'''
from os.path import join, dirname
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main, Base


path_here = dirname(dirname(__file__))
Main.include(join(path_here, "de/model.jl"))

def julia_solver(ps):
    '''
    Function receives a set of parameters,
       then return the simulation of ODE over time.
       p = [eps, w, alpha] as 1D array
    '''
    return Base.invokelatest(Main.solveODE(ps))

def julia_sol_matrix(ps):
    '''
    Function receives a set of parameters then returns the simulation of ODE at the last timepoint for all KO models,
        mimicking the experimental data.
        p = [eps, w, alpha] as 1D array
    '''
    return Base.invokelatest(Main.sol_matrix(ps))
