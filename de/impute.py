""" This file includes the functions for cross-validation based on data imputation. """
from re import L
import re
import numpy as np
from numpy import ma
from numpy.ma.extras import average
from scipy.special import expit
from scipy.special.orthogonal import legendre
from de.importData import importLINCS
from .factorization import alpha, factorizeEstimate, SparseFactorization
from .linearModel import runFitting


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

def impute(data, linear=False):
    """ Impute by repeated fitting. """
    missing = np.isnan(data)
    data = np.nan_to_num(data)

    for _ in range(50):
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

        # Fit
        if linear:
            model = runFitting(data)
        else:
            w, eta = factorizeEstimate(data)

        # Fill-in with model prediction
        if linear:
            predictt = model.predict(U)
        else:
            predictt = eta[0][:, np.newaxis] * expit(w @ U) / alpha

        dataLast = np.copy(data)
        data[missing] = predictt[missing]
        change = np.linalg.norm(data - dataLast)

    return data


def repeatImputation(data, linear=False, numIter=5):
    """ Repeat imputation and calculate the average of cost for 20 iterations. """
    coefs = []
    for _ in range(numIter):
        train_X, test_X = split_data(data)
        full_X = impute(train_X, linear)
        corr_coef = ma.corrcoef(ma.masked_invalid(full_X.flatten()), ma.masked_invalid(test_X.flatten()))
        coefs.append(corr_coef[0][1])
    print(f"average corr coef: {sum(coefs)/len(coefs)}")
    return coefs

def repeatSparsityCheck():
    cellLineList = ["A375", "A549", "HA1E", "MCF7", "PC3"]

    cellLineCoefs = []
    for x in cellLineList:
      data, _= importLINCS(x)
      coefs = repeatImputation(data, 5)
      averagecoef = sum(coefs)/len(coefs)
      cellLineCoefs.append(averagecoef)
    
    return cellLineCoefs


