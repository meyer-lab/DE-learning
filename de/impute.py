""" This file includes the functions for cross-validation based on data imputation. """
import numpy as np
from scipy.special import expit
from .factorization import alpha, factorizeEstimate

def split_data(X, n=20):
    """ Prepare the test and train data. """
    row = np.random.choice(X.shape[0], n, replace=False)
    col = np.random.choice(X.shape[1], n, replace=False)
    train_X = np.copy(X)
    test_X = np.full_like(X, np.nan)
    train_X[row, col] = np.nan
    test_X[row, col] = X[row, col]
    assert np.sum(np.isnan(train_X)) == n
    assert np.sum(np.isfinite(test_X)) == n
    return train_X, test_X


def impute(data):
    """ Impute by repeated fitting. """
    missing = np.isnan(data)
    data = np.nan_to_num(data)

    for ii in range(20):
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
