""" Methods to implement the factorization/fitting process. """

import numpy as np
import pandas as pd
from scipy.stats import gmean
from scipy.special import expit, logit
from .importData import importLINCS


alpha = 0.1


def calcW(data, eta, alphaIn):
    """Directly calculate w."""
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    B = (data * alphaIn) / eta[:, np.newaxis]
    assert np.all(np.isfinite(B))
    B = logit(np.clip(B, 0.0001, 0.9999))
    assert np.all(np.isfinite(B))
    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


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

def cellLineComparision(cellLine1, cellLine2):
    w1, eta1, annotation1 = cellLineFactorization(cellLine1)
    w2, eta2, annotation2 = cellLineFactorization(cellLine2)

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
    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _= cellLineFactorization(cellLine2)
    index_list1, index_list2 = cellLineComparision(cellLine1, cellLine2)

    w1_edited = (pd.DataFrame(w1).iloc[index_list1]).values
    w2_edited = (pd.DataFrame(w2).iloc[index_list2]).values

    num_rows_W1, num_cols_W1 = w1_edited.shape
    num_rows_W2, num_cols_W2 = w2_edited.shape


    

    if num_rows_W2 > num_rows_W1:
        removal_number = num_rows_W2 - num_rows_W1
        start_point = num_cols_W2 - removal_number
        w2_final = np.delete(w2_edited, slice(start_point, num_cols_W2, 1), 0)
        difference_matrix = w2_final - w1_edited
        norm = np.linalg.norm(difference_matrix)
        return [w1_edited, w2_final]
    
    
    elif num_rows_W1 > num_rows_W2:
        removal_number = num_rows_W1 - num_rows_W2
        start_point = num_cols_W1 - removal_number
        w1_final = np.delete(w1_edited, slice(start_point, num_cols_W1, 1), 0)
        difference_matrix = w2_edited - w1_final
        norm = np.linalg.norm(difference_matrix)
        return  [w1_final, w2_edited]
    
    
 

    




