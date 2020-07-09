import numpy as np
from scipy.optimize import least_squares
from .model import Model
from .importData import importRNAseqKO, formMatrix

def residual_fun(params, model, exp_data):
    model_data = model.sim(params)
    # Combine squared errors from all knockout models to form 1D array of residuals
    return np.sum(np.square(model_data - exp_data), axis=1)

# Import experimental data
data = importRNAseqKO()
exp_data = formMatrix(data)

# Set up ODE model
model = Model()
params = model.random_params()

# Perform least squares optimization
opt_params = least_squares(residual_fun, params, args=(model, exp_data))
