""" Methods implementing the model as a fitting process. """

import numpy as np
from jax import grad, jit
import jax.numpy as jnp
from jax.scipy.special import expit
from jax.config import config
from scipy.optimize import minimize
from .factorization import alpha, factorizeEstimate

config.update("jax_enable_x64", True)


def reshapeParams(p, nGenes, nCellLines=1):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[:(nGenes * nGenes)], (nGenes, nGenes))

    eta_list = [p[-nGenes:]]
    for i in range(1, nCellLines):
        eta = p[-nGenes*(i+1):-nGenes*(i)]
        eta_list.insert(0, eta)

    assert eta_list[0].size == w.shape[0]

    return w, eta_list


def cost(pIn, data, U=None, linear=False):
    """ Returns SSE between model and experimental RNAseq data. """
    if U is None:
<<<<<<< HEAD
        U = [np.copy(d) for d in data]
        for ii in range(len(U)):
            np.fill_diagonal(U[ii], 0.0)

    w, eta = reshapeParams(pIn, data[0].shape[0], len(data))

    costt = 0
    for i in range(len(data)):
        if linear:
            costt += jnp.linalg.norm(eta[i][:, jnp.newaxis] * (w @ U[i]) - alpha * data[i])
        else:
            costt += jnp.linalg.norm(eta[i][:, jnp.newaxis] * expit(w @ U[i]) - alpha * data[i])
    
    costt += regularize(pIn, data[0].shape[0], len(data))
    
=======
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    w, eta = reshapeParams(pIn, data.shape[0])

    if linear:
        costt = jnp.linalg.norm(eta[:, jnp.newaxis] * (w @ U) - alpha * data)
    else:
        costt = jnp.linalg.norm(eta[:, jnp.newaxis] * expit(w @ U) - alpha * data)
    costt += regularize(pIn, data.shape[0])

>>>>>>> d1ee4c1a36dbc5474ff4699b6ada90f90485430a
    return costt


def regularize(pIn, nGenes, nCellLines, strength=0.1):
    """Calculate the regularization."""
    w = reshapeParams(pIn, nGenes, nCellLines)[0]

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
    assert (res.success) or (res.nit == niter)

    return res.x
