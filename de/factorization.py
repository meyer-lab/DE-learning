""" Methods to implement the factorization/fitting process. """

from typing import Union
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import expit, logit
from .importData import importLINCS
from .fancyimpute.soft_impute import SoftImpute


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


def val_grad(w, D, eta, alpha):
    """ Calculate gradient of the cost w.r.t. w. """
    U = D.copy()
    np.fill_diagonal(U, 0.0)
    expR = expit(w @ U)
    ER = eta[:, np.newaxis] * expR

    cost = np.linalg.norm(ER - alpha * D)**2.0
    gradd = 2 * (ER - alpha * D) * (ER * (np.ones(U.shape) - expR)) @ U.T
    return cost, gradd


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


def fitW(w0, data: list, eta: list, alphaIn: float) -> np.ndarray:
    maxiter = 1000

    def costFun(x):
        wIn = np.reshape(x, w0.shape)
        gradOut = np.zeros_like(wIn)
        costOut = 0.0
        for ii in range(len(data)):
            retVal = val_grad(wIn, data[ii], eta[ii], alphaIn)
            costOut += retVal[0]
            gradOut += retVal[1]
        return costOut, gradOut.flatten()

    res = minimize(costFun, w0.flatten(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return np.reshape(res.x, w0.shape)


def calcEta(data: np.ndarray, w: np.ndarray, alphaIn: float) -> np.ndarray:
    """
    Calculate an estimate for eta based on data and current iteration of w.
    """
    assert np.all(np.isfinite(data))
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    expM = expit(w @ U)
    aData = alphaIn * data

    # Least squares with one coefficient and no intercept
    xy = np.sum(expM * aData, axis=1)
    xx = np.sum(expM * expM, axis=1)
    assert np.all(np.isfinite(xy))
    assert np.all(np.isfinite(xx))

    etta = xy / xx
    assert np.min(etta) >= 0.0
    assert np.max(etta) < 1e10
    return etta


def factorizeEstimate(data: Union[list, np.ndarray], maxiter=300, returnCost=False, returnData=False):
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

    missing = [np.isnan(d) for d in data]
    data = [SoftImpute(min_value=0.0, verbose=False).fit_transform(d) for d in data]

    w = np.zeros((data[0].shape[0], data[0].shape[0]))
    etas = [calcEta(x, w, alpha) for x in data]

    cost = costF(data, w, etas, alpha)
    linear = True

    # Use the data to try and initialize the parameters
    tq = tqdm(range(maxiter), delay=0.5)
    for ii in tq:
        etas = [calcEta(x, w, alpha) for x in data]

        if linear:
            wProposed = calcW(data, etas, alpha)
        else:
            wProposed = fitW(w, data, etas, alpha)

        for jj, dd in enumerate(data):
            U = np.copy(dd)
            np.fill_diagonal(U, 0.0)
            predictt = etas[jj][:, np.newaxis] * expit(wProposed @ U) / alpha
            data[jj][missing[jj]] = predictt[missing[jj]]

        costNew = costF(data, wProposed, etas, alpha)

        if cost - costNew > 1e-3:
            w = wProposed
            cost = costNew
        else:
            if linear:
                print(f"Switch to non-linear at {ii}.")
                linear = False
            else:
                break

        tq.set_postfix(cost=cost, refresh=False)

    if returnCost:
        return w, etas, cost

    if returnData:
        return w, etas, data

    return w, etas


def commonGenes(annots: list) -> list:
    """
    Uses annotation list to generate an array of common genes between multiple cell lines
    output
    """
    intersect = [set(ann) for ann in annots]

    intersection = intersect[0]
    for i in range(len(annots)):
        intersection = intersection & intersect[i]

    indexes = []
    for i in range(len(annots)):
        tmp = np.array([annots[i].index(x) for x in intersection], dtype=int)
        indexes.append(tmp)

    return indexes


def mergedFitting(cellLine1, cellLine2, maxiter=100):
    """
    Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix.
    """
    data1, annotation1 = importLINCS(cellLine1)
    data2, annotation2 = importLINCS(cellLine2)
    indexes = commonGenes([annotation1, annotation2])
    index_list1, index_list2 = indexes[0], indexes[1]

    # Make shared
    data1 = data1[index_list1[0:-1], :]
    data2 = data2[index_list2[0:-1], :]
    data1 = data1[:, index_list1]
    data2 = data2[:, index_list2]
    shared_data = [data1, data2]

    return factorizeEstimate(shared_data, maxiter=maxiter)
