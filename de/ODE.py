'''
This file contains ODE equations, solver and calculator for Jacobian Matrix
'''
from os.path import join, dirname
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Base


path_here = dirname(dirname(__file__))
Base.MainInclude.include(join(path_here, "de/model.jl"))


def solver(ps):
    '''
    Function receives time series and a set of parameters,
       then return the simulation of ODE.
       p = [eps, w, alpha] as 1D array
    '''
    return jl.eval('solveODE')(ps)
