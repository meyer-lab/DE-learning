""" Methods to implement the factorization/fitting process. """

from typing import Tuple, Union
import numpy as np
from tqdm import tqdm
from scipy.special import expit
from jax.scipy.special import expit as jexpit
from .importData import importLINCS
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import jacrev


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
                               * expit(w @ U[jj]) - alphaIn * data[jj])

    return cost


def calcW(data: list, eta: list, alphaIn: float, x0=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Directly calculate w.

    Calculate an estimate for w based on data and current iteration of eta
    """
    Us = [np.copy(x) for x in data]
    Ds = [alphaIn * np.copy(x) for x in data]
    for U in Us:
        np.fill_diagonal(U, 0.0)

    if x0 is None:
        x0 = np.zeros((data[0].shape[0], data[0].shape[0]))

    def cost(x):
        w = x.reshape(data[0].shape[0], data[0].shape[0])
        costt = 0.0
        for ii in range(len(Ds)):
            costt += jnp.linalg.norm(eta[ii][:, jnp.newaxis] * jexpit(w @ Us[ii]) - Ds[ii])
        return costt + 0.0001 * jnp.linalg.norm(x, ord=1)

    grd = jacrev(cost)
    optt = minimize(cost, x0, jac=grd, method="CG", options={"maxiter": 10})

    w = optt.x.reshape(data[0].shape[0], data[0].shape[0])
    return w


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


def factorizeEstimate(data: Union[list, np.ndarray], tol=1e-3, maxiter=100, returnCost=False):
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

    cost = np.inf

    # Use the data to try and initialize the parameters
    for _ in tqdm(range(maxiter), delay=1.0):
        etas = [calcEta(x, w, alpha) for x in data]
        w = calcW(data, etas, alpha, x0=w.flatten())
        costLast = cost
        cost = costF(data, w, etas, alpha)

        if (costLast - cost) < tol:
            break

    if returnCost:
        return w, etas, cost

    return w, etas


def commonGenes(annotation1, annotation2):
    """
    Uses annotation list to generate an array of common genes between two cell lines
    output
    """
    annotation1 = annotation1[0].tolist()
    annotation2 = annotation2[0].tolist()

    intersection = set(annotation1).intersection(annotation2)
    intersection_annotation = list(intersection)

    index_list1 = np.array([annotation1.index(x) for x in intersection_annotation], dtype=int)
    index_list2 = np.array([annotation2.index(x) for x in intersection_annotation], dtype=int)
    return np.sort(index_list1), np.sort(index_list2)


def MatrixSubtraction(cellLine1, cellLine2):
    """Finds the w-matrices of two different cell lines.
    Calculates the norm of the original matrices and their difference.
    """
    data1, annotation1 = importLINCS(cellLine1)
    data2, annotation2 = importLINCS(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)
    w1, _ = factorizeEstimate(data1)
    w2, _ = factorizeEstimate(data2)

    w1 = w1[index_list1, :]
    w2 = w2[index_list2, :]
    w1 = w1[:, index_list1]
    w2 = w2[:, index_list2]
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

    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)
    print(data1.shape)

    # Make shared
    data1 = data1[index_list1, :]
    data2 = data2[index_list2, :]
    data1 = data1[:, index_list1]
    data2 = data2[:, index_list2]
    shared_data = [data1, data2]

    return factorizeEstimate(shared_data)
