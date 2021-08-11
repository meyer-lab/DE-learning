from sklearn.linear_model import Lasso
import numpy as np

def runFitting(data, U=None, max_iter=300000):
    """Creates Lasso object, fits model to data"""
    
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)
    model = Lasso(max_iter=max_iter)
    model.fit(U, data)

    return model

def SSE(data, U, model):
    """Calculates the SSE of the model"""

    p = model.predict(U)
    diff = np.absolute(data - p)
    square = np.power(diff,2)
    sum_errors = np.sum(square)
    
    return sum_errors