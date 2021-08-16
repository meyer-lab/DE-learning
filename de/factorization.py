""" Methods to implement the factorization/fitting process. """

import numpy as np
import pandas as pd
from scipy.stats import gmean
from scipy.special import expit, logit
from .importData import 


alpha = 0.1


def calcW(data, eta, alphaIn):
    """Directly calculate w."""
    if isinstance(data, np.ndarray):
        data = [data]

    U = None
    B = None

    for i, x in enumerate(data):
        U1 = np.copy(x)
        np.fill_diagonal(U1, 0.0)
        B1 = (x * alphaIn) / eta[i][:, np.newaxis]
        assert np.all(np.isfinite(B1))
        B1 = logit(np.clip(B1, 0.0001, 0.9999))
        assert np.all(np.isfinite(B1))

        if B is None:
            U = U1
            B = B1
        else:
            U = np.concatenate((U, U1), axis=1)
            B = np.concatenate((B, B1), axis=1)

    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data, w, alphaIn):
    """Directly calculate eta."""
    eta = (alphaIn * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)


def factorizeEstimate(data, tol=1e-9, maxiter=20):
    """ Initialize the parameters based on the data. """
    assert maxiter > 0
    # TODO: Add tolerance for termination.
    if isinstance(data, np.ndarray):
        data = [data]

    w = np.zeros((data[0].shape[0], data[0].shape[0]))
    # Make the U matrix
    U = [np.copy(d) for d in data]
    for ii in range(len(U)):
        np.fill_diagonal(U[ii], 0.0)

    costLast = np.inf

    # Use the data to try and initialize the parameters
    for ii in range(maxiter):
        etas = [calcEta(x, w, alpha) for x in data]
        for eta in etas:
            assert np.all(np.isfinite(eta))
            assert eta.shape == (data[0].shape[0], )

        w = calcW(data, etas, alpha)
        assert np.all(np.isfinite(w))
        assert w.shape == (data[0].shape[0], data[0].shape[0])

        cost = 0
        for jj in range(len(data)):
            cost += np.linalg.norm(etas[jj][:, np.newaxis] * expit(w @ U[jj]) - alpha * data[jj])

        if ii > 3 and (costLast - cost) < tol:
            # TODO: I believe the cost should be strictly decreasing, so look into this.
            break

        costLast = cost

    return w, etas


def cellLineFactorization(cellLine):
    """ Import a cell line, fit the model, and return the result. """
    data, annotation = importLINCS(cellLine)
    w, eta = factorizeEstimate(data)
    return w, eta, annotation[0].tolist()


def commonGenes(annotation1, annotation2):
    """Uses annotation list to generate an array of common genes between two cell lines"""

    annotation1 = annotation1[0].tolist()
    annotation2 = annotation2[0].tolist()

    intersection = set(annotation1).intersection(annotation2)
    intersection_annotation = list(intersection)

    index_list1 = [annotation1.index(x) for x in intersection_annotation]
    index_list2 = [annotation2.index(x) for x in intersection_annotation]

    index_list1.sort()
    index_list2.sort()
    return index_list1, index_list2


def MatrixSubtraction(cellLine1, cellLine2):
    """Subtracts the w-matrices of two different cell lines and subtracts them.
    Then, it calculates the norm of the original matrices as well as difference matrix"""
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)
    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)

    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    np.random.shuffle(w1)
    np.random.shuffle(w2)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    w1_df = pd.DataFrame(w1)
    w2_df = pd.DataFrame(w2)

    w1_edited = w1_df.iloc[index_list1, index_list1]
    w2_edited = w2_df.iloc[index_list2, index_list2]

    w1_final = w1_edited.values
    w2_final = w2_edited.values

    difference_matrix = w2_final - w1_final
    diff_norm = np.linalg.norm(difference_matrix)
    return norm1, norm2, diff_norm, w1_final, w2_final


def mergedFitting(cellLine1, cellLine2):
    """Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix."""
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)

    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)
    data1_df = pd.DataFrame(data1)
    data2_df = pd.DataFrame(data2)
    data1_edited = data1_df.iloc[index_list1, index_list1]
    data2_edited = data2_df.iloc[index_list2, index_list2]
    data1_final = data1_edited.values
    data2_final = data2_edited.values
    shared_data = [data1_final, data2_final]

    w_shared, eta_list = factorizeEstimate(shared_data)

    return w_shared, eta_list
