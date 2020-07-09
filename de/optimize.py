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

# Import experimental data
data = importRNAseqKO()
exp = formMatrix(data)

# Set up ODE model
m = Model()
p = model.random_params()

# Perform least squares optimization
#opt_params = least_squares(residual_fun, params, args=(model, exp_data))
