""" Methods implementing the model as a fitting process. """

import numpy as np
from jax import grad, jit
import jax.numpy as jnp
from jax.scipy.special import expit
from jax.config import config
from scipy.optimize import minimize
from .factorization import alpha, factorizeEstimate

config.update("jax_enable_x64", True)


def reshapeParams(p, nGenes):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[:(nGenes * nGenes)], (nGenes, nGenes))
    eta = p[-nGenes:]

    assert eta.size == w.shape[0]

    return w, eta


def cost(pIn, data, U=None, linear=False):
    """ Returns SSE between model and experimental RNAseq data. """
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    w, eta = reshapeParams(pIn, data.shape[0])

    if linear:
        costt = jnp.linalg.norm(eta[:, jnp.newaxis] * (w @ U) - alpha * data)
    else:
        costt = jnp.linalg.norm(eta[:, jnp.newaxis] * expit(w @ U) - alpha * data)
    costt += regularize(pIn, data.shape[0])

    return costt


def regularize(pIn, nGenes, strength=0.1):
    """Calculate the regularization."""
    w = reshapeParams(pIn, nGenes)[0]

    ll = jnp.linalg.norm(w, ord=1)
    ll += jnp.linalg.norm(w.T @ w - jnp.identity(w.shape[0]))
    return strength * ll


def runOptim(data, niter=2000, disp=0, linear=False):
    """ Run the optimization. """
    # TODO: Add bounds to fitting.
    w, eps = factorizeEstimate(data)
    x0 = np.concatenate((w.flatten(), eps))

    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    cost_grad = jit(grad(cost, argnums=0), static_argnums=(3,))

    def cost_GF(*args):
        outt = cost_grad(*args)
        return np.array(outt)

    res = minimize(cost, x0, args=(data, U, linear), method="L-BFGS-B", jac=cost_GF, options={"maxiter": niter, "disp": disp})
    assert (res.success) or (res.nit == niter)

    return res.x
