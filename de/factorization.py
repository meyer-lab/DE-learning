""" Methods to implement the factorization/fitting process. """

from typing import Union
import numpy as np
from tqdm import tqdm
from scipy.special import expit, logit
from .importData import importLINCS


alpha = 0.1


def costF(data: list, w, etas: list, alphaIn):
    """ Calculate the fitting cost. """
    assert len(data) == len(etas)
    assert w.shape == (data[0].shape[0], data[0].shape[0])
    assert np.all(np.isfinite(w))
    for eta in etas:
        assert np.all(np.isfinite(eta))
        assert eta.shape == (data[0].shape[0], )
    # Make the U matrix
    U = [np.copy(d) for d in data]
    for ii in range(len(U)):
        np.fill_diagonal(U[ii], 0.0)
    cost = 0.0
    for jj in range(len(data)):
        cost += np.linalg.norm(etas[jj][:, np.newaxis]
                               * expit(w @ U[jj]) - alphaIn * data[jj])**2.0
    return cost


def calcW(data: list, eta: list, alphaIn: float) -> np.ndarray:
    """
    Directly calculate w.
    Calculate an estimate for w based on data and current iteration of eta
    """
    for i, x in enumerate(data):
        U1 = np.copy(x)
        np.fill_diagonal(U1, 0.0)
        B1 = (x * alphaIn) / eta[i][:, np.newaxis]
        B1 = logit(np.clip(B1, 0.001, 0.999))

        if i == 0:
            U = U1
            B = B1
        else:
            U = np.concatenate((U, U1), axis=1)
            B = np.concatenate((B, B1), axis=1)

    return np.linalg.lstsq(U.T, B.T, rcond=None)[0].T


def calcEta(data: np.ndarray, w: np.ndarray, alphaIn: float) -> np.ndarray:
    """
    Calculate an estimate for eta based on data and current iteration of w.
    """
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    expM = expit(w @ U)
    aData = alphaIn * data

    # Least squares with one coefficient and no intercept
    xy = np.sum(expM * aData, axis=1)
    xx = np.sum(expM * expM, axis=1)

    etta = xy / xx
    assert np.min(etta) >= 0.0
    assert np.max(etta) < 1e10
    return etta


def factorizeEstimate(data: Union[list, np.ndarray], maxiter=100, returnCost=False):
    """
    Iteravely solve for w and eta list based on the data.
    :param data: matrix or list of matrices representing a cell line's gene expression interactions with knockdowns
    :type data: Array or Array List
    param tol: the minimum difference between two cost iteration values necessary to stop factorization process
    :type tol: float
    param maxiter: maximum amount of iterations for factorization process, given no tolerance break
    :type maxiter: int
    output w: finalized matrix representing gene-to-gene pertubation effects for either a singular cell line or multiple cell lines
    :type w: Array
    :output etas: finalized list of vectors representing overall perturbation effect of genes in each cell line
    :type etas: Array List
    """
    assert maxiter > 0
    if isinstance(data, np.ndarray):
        data = [data]

    w = np.zeros((data[0].shape[0], data[0].shape[0]))
    etas = [calcEta(x, w, alpha) for x in data]

    cost = costF(data, w, etas, alpha)

    # Use the data to try and initialize the parameters
    tq = tqdm(range(maxiter), delay=0.5)
    for _ in tq:
        wProposed = calcW(data, etas, alpha)
        etasProposed = [calcEta(x, w, alpha) for x in data]
        costProposed = costF(data, wProposed, etasProposed, alpha)

        if costProposed < cost:
            cost = costProposed
            etas = etasProposed
            w = wProposed
        else:
            break

        tq.set_postfix(cost=cost, refresh=False)

    if returnCost:
        return w, etas, cost

    return w, etas


def commonGenes(ann1, ann2):
    """
    Uses annotation list to generate an array of common genes between two cell lines
    output
    """
    intersection = list(set(ann1) & set(ann2))

    idx1 = np.array([ann1.index(x) for x in intersection], dtype=int)
    idx2 = np.array([ann2.index(x) for x in intersection], dtype=int)
    return idx1, idx2


def MatrixSubtraction(cellLine1, cellLine2):
    """Finds the w-matrices of two different cell lines.
    Calculates the norm of the original matrices and their difference.
    """
    data1, annotation1 = importLINCS(cellLine1)
    data2, annotation2 = importLINCS(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)
    w1, _ = factorizeEstimate(data1)
    w2, _ = factorizeEstimate(data2)

    w1 = w1[index_list1, index_list1]
    w2 = w2[index_list2, index_list2]
    assert w1.shape == w2.shape
    assert w1.shape == (index_list1.size, index_list1.size)

    norm1 = np.linalg.norm(w1)
    print(f"Norm1: {norm1}")
    norm2 = np.linalg.norm(w2)
    print(f"Norm2: {norm2}")
    diff_norm = np.linalg.norm(w2 - w1)
    print(f"Difference norm: {diff_norm}")

    w1shuff = w1.copy()
    w2shuff = w2.copy()
    np.random.shuffle(w1shuff)
    np.random.shuffle(w2shuff)
    shufnorm = np.linalg.norm(w2shuff - w1shuff)
    print(f"Shuffled norm: {shufnorm}")
    return norm1, norm2, diff_norm, shufnorm, w1, w2


def mergedFitting(cellLine1, cellLine2):
    """
    Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix.
    """
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)
    idx1 = index_list1.copy()
    idx2 = index_list2.copy()
    np.concatenate(idx1, (len(annotation1) + 1)) # include the control
    np.concatenate(idx2, (len(annotation2) + 1)) # include the control

    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)

    # Make shared
    data1 = data1[index_list1, :]
    data2 = data2[index_list2, :]
    data1 = data1[:, idx1]
    data2 = data2[:, idx2]
    shared_data = [data1, data2]

    return factorizeEstimate(shared_data)


def grad(w, D, eta, alpha):
    """Calculate gradient of the cost w.r.t. w. """
    U = D.copy()
    np.fill_diagonal(U, 0.0)
    def d_expit(U, w):
        return (expit(w @ U) * (np.ones((U.shape[0], U.shape[1])) - expit(w @ U))) @ U.T

    first = np.trace((eta.T * d_expit(U, w).T) @ (eta[:, np.newaxis] * expit(w @ U)))
    second = np.trace((eta.T * expit(w@U).T) @ (eta * d_expit(U, w)))
    third = -2 * alpha * np.trace((eta * d_expit(U, w) @ D))
    return first + second + third
