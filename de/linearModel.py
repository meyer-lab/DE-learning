from sklearn.linear_model import Lasso
import numpy as np
from .fancyimpute.soft_impute import SoftImpute


def runFitting(data, U=None, alpha=1.0):
    """ Creates Lasso object, fits model to data. """

    missing = np.isnan(data)
    data = SoftImpute(min_value=0.0, verbose=False).fit_transform(data)

    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    model = Lasso(max_iter=300000, alpha=alpha)
    model.fit(U, data)

    predictt = model.predict(U)
    dataLast = np.copy(data)
    data[missing] = predictt[missing]

    return [data]
