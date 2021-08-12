""" This file includes the functions for cross-validation based on data imputation. """
import numpy as np
from scipy.special import expit
from .factorization import alpha, factorizeEstimate

def impute(data):
    """ Impute by repeated fitting. """
    missing = np.isnan(data)
    data = np.nan_to_num(data)

    for ii in range(10):
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)
        data_last = np.copy(data)

        # Fit
        w, eta = factorizeEstimate(data)

        # Fill-in with model prediction
        predictt = eta[0][:, np.newaxis] * expit(w @ U) / alpha
        data[missing] = predictt[missing]

        print(np.linalg.norm(data - data_last))

    return data
