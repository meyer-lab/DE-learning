"""
File contains function for finding residual between model data and experimental RNAseq data and command to perform scipy.optimize.least_squares
"""
from scipy.optimize import least_squares
from .model import Model
from .importData import importRNAseqKO, formMatrix

def residual_fun(params, model, exp_data):
    """Function takes in optimizable parameters, Model() instance, and experimental data matrix; returns 1D array of residuals."""
    model_data = model.sim(params)
    return model_data.flatten()-exp_data.flatten()

def lsq(params, model, exp_data, residual_fun):
    """
    Parameters:
    - params: 1D array of eps, w, and alpha values to be optimized
    - model: Model() object to be simulated
    - exp_data: 2D (83x84) array of RNAseq KO data
    - residual_fun: function that returns 1D array of residuals
    Returns 1D array of optimized parameters and minimized value of Cost function using scipy.optimize.least_squares
    """
    opt_params, cost = least_squares(residual_fun, params, args=(model, exp_data))
    return opt_params, cost
