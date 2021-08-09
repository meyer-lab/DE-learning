from sklearn.linear_model import Lasso
from .importData import ImportMelanoma, importLINCS
import numpy as np

def runFitting(data, U=None, max_iter=300000):
    #cellLines = ["A375", "A549", "HA1E", "HT29", "MCF7", "PC3"]
    
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    model = Lasso(max_iter=max_iter)
    model.fit(U, data)

    p = model.predict(U)

    #cost = SSE(data, U, model)
    
    #np.savetxt(f"{cellLine}.csv", model.coef_, delimiter=",")

    return model.coef_, p

def SSE(data, U, model):

    p = model.predict(U)
    diff = np.absolute(data - p)
    
    square = np.power(diff,2)
    sum_errors = np.sum(square)
    
    return sum_errors
