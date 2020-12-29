from jax import grad, jit
import jax.numpy as jnp
from jax.nn import softplus


def reshapeParams(p):
    """Reshape a vector of parameters into the variables we know."""
    w = jnp.reshape(p[1:6889], (83, 83))
    eta = p[6890:6972]

    assert eta.size == w.shape[0]

    return w, softplus(eta)


def cost(pIn, data, U = None):
    """ Returns SSE between model and experimental RNAseq data. """
    if U is None:
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)

    w, eta = reshapeParams(pIn)
    costt = jnp.linalg.norm(eta * (1 + jnp.tanh(w @ U)) - alpha * data)
    costt += regularize(pIn)

    return costt


cost_grad = grad(cost, argnums=0)


def regularize(pIn, strength = 1.0):
    """Calculate the regularization."""
    w = reshapeParams(pIn)[0]

    ll = jnp.linalg.norm(w, ord=1)
    ll += jnp.linalg.norm(w.T @ w - jnp.identity(w.shape[0]))
    return strength * ll


def costG(x, data):
    """ Cost function gradient. """
    U = jnp.copy(data)
    jnp.fill_diagonal(U, 0.0)
    G = cost_grad(x, data, U)
    return G

