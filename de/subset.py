""" Gene subset analysis. """

from .importData import ImportMelanoma
from .factorization import factorizeEstimate
from scipy.special import expit
import pandas as pd

alpha = 0.1

def import_factorize():
    """ Only analyze FOSL1, FOXC2, JUN, JUNB from the melanoma dataset. """
    mel = ImportMelanoma()
    m = mel[['FOSL1', 'FOXC2', 'JUN', 'JUNB', 'neg']].loc[['FOSL1', 'FOXC2', 'JUN', 'JUNB']] # keep control

    return factorizeEstimate(m)

# plot network
def plot_net(ax):
    """ plot the network of gene subset. """
    Network(w, ax)

# plot ODE for these 4 genes
def solver(w, eta, ts):
    '''Function receives time series and a set of parameters,
       then return the simulation of ODE.
    '''

    x0 = eta / alpha
    sol = odeint(ODE, x0, ts, args=(eta, w))
    return sol

def ODE(y, eta, w):
    '''The ODE system:
    Parameters = eps: Value that bound the saturation effect
                 w: Interaction between components
                 alpha: Degradation rate
    '''

    U = y.copy()
    U = np.fill_diagonal(U, 0.0)
    return eps * expit(w @ U) - alpha * y
