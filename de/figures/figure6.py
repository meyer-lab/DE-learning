""" Gene subset analysis. """

from ..importData import ImportMelanoma
from ..factorization import factorizeEstimate
from ..graph import Network
from scipy.special import expit
from .common import subplotLabel, getSetup
import numpy as np
import pandas as pd
from scipy.integrate import odeint

alpha = 0.1

def import_factorize():
    """ Only analyze FOSL1, FOXC2, JUN, JUNB from the melanoma dataset. """
    mel = ImportMelanoma()
    m = mel[['FOSL1', 'FOXC2', 'JUN', 'JUNB', 'neg']].loc[['FOSL1', 'FOXC2', 'JUN', 'JUNB']] # keep control

    return factorizeEstimate(m.to_numpy())


# plot ODE for these 4 genes
def solver(w, eta, ts):
    '''Function receives time series and a set of parameters,
       then return the simulation of ODE.
    '''

    x0 = eta / alpha
    sol = odeint(ODE, x0, ts, args=(eta, w))
    return sol

def ODE(y, _, eta, w):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
    '''


    return eta * expit(np.dot(w, y)) - alpha * y

def makeFigure():

    ax, f = getSetup((6, 3), (1, 2))
    w, eta = import_factorize()

    # plot network
    Network(pd.DataFrame(w, index=['FOSL1', 'FOXC2', 'JUN', 'JUNB'], columns=['FOSL1', 'FOXC2', 'JUN', 'JUNB']), ax[0])

    # plot over time
    ts = np.arange(0.0, 48, 4)
    sol = solver(w, eta[0], ts)
    ax[0].axis('off')
    ax[0].set_title("gene-gene network")

    ax[1].plot(ts, sol)
    ax[1].legend(['FOSL1', 'FOXC2', 'JUN', 'JUNB'])
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("expr. level")
    return f
