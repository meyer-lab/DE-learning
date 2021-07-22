""" Methods implementing the model as a fitting process. """

import numpy as np
import pandas as pd
from jax import grad, jit
import jax.numpy as jnp
from jax.scipy.special import expit
from jax.config import config
from scipy.optimize import minimize
from .factorization import alpha, factorizeEstimate, cellLineComparison
from .importData import importLINCS

config.update("jax_enable_x64", True)


def reshapeParams(p, nGenes):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[:(nGenes * nGenes)], (nGenes, nGenes))
    nCellLines = len(p)/nGenes - nGenes

    eta_list = [p[-nGenes:]]
    for i in range(1, int(nCellLines)):
        eta = p[-nGenes*(i+1):-nGenes*(i)]
        eta_list.insert(0, eta)

    assert len(eta_list[0]) == w.shape[0]

    return w, eta_list


def cost(pIn, data, U=None, linear=False):
    """ Returns SSE between model and experimental RNAseq data. """
    if isinstance(data, np.ndarray):
        data = [data]
    if U is None:
        U = [np.copy(d) for d in data]
        for ii in range(len(U)):
            np.fill_diagonal(U[ii], 0.0)

    w, eta = reshapeParams(pIn, data[0].shape[0])

    costt = 0
    for i in range(len(data)):
        if linear:
            costt += jnp.linalg.norm(eta[i][:, jnp.newaxis] * (w @ U[i]) - alpha * data[i])
        else:
            costt += jnp.linalg.norm(eta[i][:, jnp.newaxis] * expit(w @ U[i]) - alpha * data[i])
    
    costt += regularize(pIn, data[0].shape[0])
    
    return costt


def regularize(pIn, nGenes, strength=0.1):
    """Calculate the regularization."""
    w = reshapeParams(pIn, nGenes)[0]

    ll = jnp.linalg.norm(w, ord=1)
    ll += jnp.linalg.norm(w.T @ w - jnp.identity(w.shape[0]))
    return strength * ll


def runOptim(data, niter=2000, disp=0, linear=False):
    """ Run the optimization. """
    if isinstance(data, np.ndarray):
        data = [data]
        
    w, eps = factorizeEstimate(data[0])
    x0 = np.concatenate((w.flatten(), eps))
    for ii in range(1, len(data)):
        x0 = np.concatenate((x0, eps))


    U = [np.copy(d) for d in data]
    for ii in range(len(U)):
        np.fill_diagonal(U[ii], 0.0)
    cost_grad = jit(grad(cost, argnums=0), static_argnums=(3,))

    def cost_GF(*args):
        outt = cost_grad(*args)
        return np.array(outt)

    res = minimize(cost, x0, args=(data, U, linear), method="L-BFGS-B", jac=cost_GF, options={"maxiter": niter, "disp": disp})
    #assert (res.success) or (res.nit == niter)
        
    return res.x

def mergedFitting(cellLine1, cellLine2):
    """Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix."""
    index_list1, index_list2 = cellLineComparison(cellLine1, cellLine2)
    
    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)
    data1_df = pd.DataFrame(data1)
    data2_df = pd.DataFrame(data2)
    data1_edited = data1_df.iloc[index_list1, index_list1]
    data2_edited = data2_df.iloc[index_list2, index_list2]
    data1_final = data1_edited.values
    data2_final = data2_edited.values
    shared_data = [data1_final, data2_final]

    p = runOptim(shared_data)

    w_shared, eta_list = reshapeParams(p, shared_data[0].shape[0])
    cost_1 = cost(np.concatenate((w_shared.flatten(), eta_list[0])), data1_final)
    cost_2 = cost(np.concatenate((w_shared.flatten(), eta_list[1])), data2_final)
    cost_shared = cost(w_shared, [data1_final, data2_final])

    return cost_1, cost_2, cost_shared
