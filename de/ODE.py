'''
This file contains ODE equation solvers from Julia
'''
# pylint: disable=no-name-in-module, wrong-import-position
from os.path import join, dirname
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main


Main.include(join(dirname(dirname(__file__)), "de/model.jl"))


def julia_solver(ps):
    '''
    Function receives a set of parameters,
       then return the simulation of ODE over time.
       p = [eps, w, alpha] as 1D array
    '''
    return Main.solveODE(ps)

def julia_sol_matrix(ps):
    '''
    Function receives a set of parameters then returns the simulation of ODE at the last timepoint for all KO models,
        mimicking the experimental data.
        p = [eps, w, alpha] as 1D array
    '''
    return Main.sol_matrix(ps)
