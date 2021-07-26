""" Methods to implement the factorization/fitting process. """

import numpy as np
import pandas as pd
from scipy.stats import gmean
from scipy.special import expit, logit
from .importData import importLINCS


alpha = 0.1


def calcW(data, eta, alphaIn):
    """Directly calculate w."""

    w = None

    
    if (all(isinstance(i, np.ndarray) for i in data)):
      for x in data:
        U1 = np.copy(x)
        np.fill_diagonal(U1, 0.0)
        B = (x * alphaIn) / eta[:, np.newaxis]
        assert np.all(np.isfinite(B))
        B = logit(np.clip(B, 0.0001, 0.9999))
        assert np.all(np.isfinite(B))
        np.concatenate(w, np.linalg.lstsq(U1.T, B.T, rcond=None)[0].T)
    else:
      U1 = np.copy(data)
      np.fill_diagonal(U1, 0.0)
      B = (data * alphaIn) / eta[:, np.newaxis]
      assert np.all(np.isfinite(B))
      B = logit(np.clip(B, 0.0001, 0.9999))
      assert np.all(np.isfinite(B))
      np.concatenate(w, np.linalg.lstsq(U1.T, B.T, rcond=None)[0].T)

    return w


def calcEta(data, w, alphaIn):
    """Directly calculate eta."""
    eta = (alphaIn * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    return gmean(eta, axis=1)


def factorizeEstimate(data, tol=1e-9, maxiter=10000):
    """ Initialize the parameters based on the data. """
    assert maxiter > 0
    # TODO: Add tolerance for termination.
    w = np.zeros((data.shape[0], data.shape[0]))
    # Make the U matrix
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    # Use the data to try and initialize the parameters
    for ii in range(maxiter):
        eta = calcEta(data, w, alpha)
        assert np.all(np.isfinite(eta))
        assert eta.shape == (data.shape[0], )
        w = calcW(data, eta, alpha)
        assert np.all(np.isfinite(w))

        assert w.shape == (data.shape[0], data.shape[0])

        cost = np.linalg.norm(eta[:, np.newaxis] * expit(w @ U) - alpha * data)

        if ii > 10 and (costLast - cost) < tol:
            # TODO: I believe the cost should be strictly decreasing, so look into this.
            break

        costLast = cost

    return w, eta


def cellLineFactorization(cellLine):
    """ Import a cell line, fit the model, and return the result. """
    data, annotation = importLINCS(cellLine)
    w, eta = factorizeEstimate(data)
    return w, eta, annotation[0].tolist()


def cellLineComparison(cellLine1, cellLine2):
    """Uses annotation list to generate an array of common genes between two cell lines"""
    _, _, annotation1 = cellLineFactorization(cellLine1)
    _, _, annotation2 = cellLineFactorization(cellLine2)

    line1_as_set = set(annotation1)
    intersection = line1_as_set.intersection(annotation2)
    intersection_annotation = list(intersection)

    index_list1 = []
    index_list2 = []

    for x in intersection_annotation:
        index_value1 = annotation1.index(x)
        index_list1.append(index_value1)

    for x in intersection_annotation:
        index_value2 = annotation2.index(x)
        index_list2.append(index_value2)

    index_list1.sort()
    index_list2.sort()
    return index_list1, index_list2


def MatrixSubtraction(cellLine1, cellLine2):
    """Subtracts the w-matrices of two different cell lines and subtracts them.
    Then, it calculates the norm of the original matrices as well as difference matrix"""
    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    index_list1, index_list2 = cellLineComparison(cellLine1, cellLine2)

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
