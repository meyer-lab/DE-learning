import numpy as np


def logisticF(p, data):
    """ Generalized logistic function. """
    return np.power(1 + p[0]*np.exp(-p[1]*data), -p[2])


def invlogF(p, invdata):
    """ Inverse generalized logistic function. """
    return -(1.0 / p[1]) * np.log((np.power(invdata, -1.0 / p[2]) - 1) / p[0])