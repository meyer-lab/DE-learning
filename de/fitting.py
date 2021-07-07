""" Methods implementing the model as a fitting process. """

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from jax import grad, jit
import jax.numpy as jnp
from jax.scipy.special import expit
from jax.config import config
from scipy.optimize import minimize
from .factorization import alpha, factorizeEstimate
from .importData import importLINCS

config.update("jax_enable_x64", True)


def reshapeParams(p, nGenes):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[:(nGenes * nGenes)], (nGenes, nGenes))
    eta = p[-nGenes:]

    assert eta.size == w.shape[0]

    return w, eta


def cost(pIn, data, U=None):
    """ Returns SSE between model and experimental RNAseq data. """
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    w, eta = reshapeParams(pIn, data.shape[0])
    costt = jnp.linalg.norm(eta[:, jnp.newaxis] * expit(w @ U) - alpha * data)
    costt += regularize(pIn, data.shape[0])

    return costt


def regularize(pIn, nGenes, strength=0.1):
    """Calculate the regularization."""
    w = reshapeParams(pIn, nGenes)[0]

    ll = jnp.linalg.norm(w, ord=1)
    ll += jnp.linalg.norm(w.T @ w - jnp.identity(w.shape[0]))
    return strength * ll


def runOptim(data, niter=2000, disp=0):
    """ Run the optimization. """
    # TODO: Add bounds to fitting.
    w, eps = factorizeEstimate(data)
    x0 = np.concatenate((w.flatten(), eps))

    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    cost_grad = jit(grad(cost, argnums=0))

    def cost_GF(*args):
        outt = cost_grad(*args)
        return np.array(outt)

    res = minimize(cost, x0, args=(data, U), method="L-BFGS-B", jac=cost_GF, options={"maxiter": niter, "disp": disp})
    assert (res.success) or (res.nit == niter)

    return res.x

def cellLineFactorization(cellLine):
    """ Import a cell line, fit the model, and return the result. """
    data, annotation = importLINCS(cellLine)
    w, eta = runOptim(data)
    return w, eta, annotation[0].tolist()

def cellLineComparison(cellLine1, cellLine2):
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
    index_list1, index_list2 = cellLineComparison(cellLine1, cellLine2)

    w1_df = pd.DataFrame(w1)
    w2_df = pd.DataFrame(w2)

    w1_edited = w1_df.iloc[index_list1, index_list1]
    w2_edited = w2_df.iloc[index_list2, index_list2]

    w1_final = w1_edited.values
    w2_final = w2_edited.values

    difference_matrix = w2_final - w1_final
    norm = np.linalg.norm(difference_matrix)
    return difference_matrix, norm, w1_final, w2_final

def PearsonWMatrix(cellLine1, cellLine2):
    _, _, w1, w2 = MatrixSubtraction(cellLine1, cellLine2)
    pearson = np.corrcoef(w1.flatten(), w2.flatten())
    return pearson

def SpearmanWMatrix(cellLine1, cellLine2):
    _, _, w1, w2 = MatrixSubtraction(cellLine1, cellLine2)
    spearman = spearmanr(w1.flatten(), w2.flatten())
    return spearman
    