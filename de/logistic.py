import jax.numpy as jnp


def logisticF(p, data):
    """ Generalized logistic function. """
    return jnp.power(1 + p[0]*jnp.exp(-p[1]*data), -p[2])


def invlogF(p, invdata):
    """ Inverse generalized logistic function. """
    return -(1.0 / p[1]) * jnp.log((jnp.power(invdata, -1.0 / p[2]) - 1) / p[0])
