import numpy as np
from jax import grad, jit
import jax.numpy as jnp
from .logistic import logisticF
from jax.config import config
from scipy.optimize import minimize
from .factorization import alpha, factorizeEstimate

config.update("jax_enable_x64", True)


def reshapeParams(p, nGenes):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[:(nGenes * nGenes)], (nGenes, nGenes))
    eta = p[(nGenes * nGenes):-3]
    pOut = p[-3::]

    assert eta.size == w.shape[0]
    assert pOut.size == 3

    return w, eta, pOut


def cost(pIn, data, U):
    """ Returns SSE between model and experimental RNAseq data. """
    w, eta, p = reshapeParams(pIn, data.shape[0])
    costt = jnp.linalg.norm(eta[:, jnp.newaxis] * logisticF(p, w @ U) - alpha * data)
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
    (w, eta, p), _ = factorizeEstimate(data)
    x0 = np.concatenate((w.flatten(), eta, p))
    assert False

    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    cost_grad = jit(grad(cost, argnums=0))

    def hvp(x, v, data, U):
        return grad(lambda x: jnp.vdot(cost_grad(x, data, U), v))(x)

    res = minimize(cost, x0, args=(data, U), method="trust-constr", jac=cost_grad, hessp=hvp, options={"maxiter": niter, "disp": disp})
    assert (res.success) or (res.nit == niter)

    return res.x
