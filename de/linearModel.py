from sklearn.linear_model import Lasso
from .importData import ImportMelanoma, importLINCS
from .factorization import cross_val
import numpy as np

def runFitting(cellLine=None, max_iter=300000):
    #cellLines = ["A375", "A549", "HA1E", "HT29", "MCF7", "PC3"]
    if cellLine is None:
        data = ImportMelanoma()
    else:
        data, annotation = importLINCS(cellLine)
    
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    model = Lasso(max_iter=max_iter)
    model.fit(U, data)

    cost = SSE(data, U, model)
    
    #np.savetxt(f"{cellLine}.csv", model.coef_, delimiter=",")

    return model.coef_, cost

def SSE(data, U, model):

    p = model.predict(U)
    diff = np.absolute(data - p)
    
    square = np.power(diff,2)
    sum_errors = np.sum(square)
    
    return sum_errors

def runCV(model):
    