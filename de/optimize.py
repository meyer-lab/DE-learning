"""
File contains function for finding residual between model data and experimental RNAseq data and command to perform scipy.optimize.least_squares
"""
from scipy.optimize import least_squares
from .model import Model

def residual_fun(params, beta_in, model, exp_data):
    """Function takes in optimizable parameters, Model() instance, and experimental data matrix; returns 1D array of residuals."""
    model_data = model.sim(params, beta_in)
    return model_data.flatten()-exp_data.flatten()

def lsq(params, model, beta_in, exp_data, fun):
    """
    Parameters:
    - params: 1D array of eps, w, and alpha values to be optimized
    - model: Model() object to be simulated
    - exp_data: 2D (83x84) array of RNAseq KO data
    - fun: function that returns 1D array of residuals
    Returns 1D array of optimized parameters and minimized value of Cost function using scipy.optimize.least_squares
    """
    opt_params, cost = least_squares(residual_fun, params, args=(beta_in, model, exp_data))
    return opt_params, cost
