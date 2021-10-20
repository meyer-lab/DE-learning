""" Methods to implement the factorization/fitting process. """

from typing import Union
import numpy as np
from .importData import importLINCS
import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy.special import expit as jexpit
from scipy.optimize import minimize
from jax.config import config
from tqdm import tqdm


config.update("jax_enable_x64", True)

alpha = 0.1


def costF(data: list, w):
    """ Calculate the fitting cost. """
    assert w.shape == (data[0].shape[0], data[0].shape[0])
    assert jnp.all(jnp.isfinite(w))

    # Make the U matrix
    U = [np.copy(d) for d in data]
    for ii in range(len(U)):
        np.fill_diagonal(U[ii], 0.0)
    cost = 0.0
    etas = []
    for jj in range(len(data)):
        expM = jexpit(w @ U[jj])
        aData = alpha * data[jj]

        # Calc eta
        # Least squares with one coefficient and no intercept
        xy = jnp.sum(expM * aData, axis=1)
        xx = jnp.sum(expM * expM, axis=1)

        etta = xy / xx
        etta = jnp.clip(etta, 0.0, 1e12)
        etas.append(etta)

        cost += jnp.linalg.norm(etta[:, jnp.newaxis] * expM - alpha * data[jj])**2.0

    for eta in etas:
        assert jnp.all(jnp.isfinite(eta))
        assert eta.shape == (data[0].shape[0], )

    return cost, etas


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

    w0 = np.zeros((data[0].shape[0], data[0].shape[0]))

    def costFun(x):
        wIn = jnp.reshape(x, w0.shape)
        return costF(data, wIn)[0]

    valGrad = value_and_grad(costFun)

    def v_g(x):
        a, b = valGrad(x)
        return a, np.array(b, order="C", copy=True)

    if costFun(w0.flatten()) > 1e-6:
        with tqdm(total=maxiter) as pbar:
            def verbose(xk):
                pbar.update(1)
                pbar.set_postfix(cost=costFun(xk), refresh=False)

            res = minimize(v_g, w0.flatten(), jac=True, method="L-BFGS-B", callback=verbose, options={"maxiter": maxiter})
        w = np.reshape(res.x, w0.shape)
    else:
        w = w0

    cosst, etas = costF(data, w)

    if returnCost:
        return w, etas, cosst

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
