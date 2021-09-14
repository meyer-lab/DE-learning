from sklearn.linear_model import Lasso
import numpy as np

def runFitting(data, U=None, alpha=1.0):
    """ Creates Lasso object, fits model to data. """
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    model = Lasso(max_iter=300000, alpha=alpha)
    model.fit(U, data)

    return model
