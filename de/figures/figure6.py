""" Gene subset analysis. """

from ..importData import ImportMelanoma, importLINCS
from ..factorization import factorizeEstimate
from ..graph import Network
from scipy.special import expit
from .common import subplotLabel, getSetup
import numpy as np
import pandas as pd
from scipy.integrate import odeint

alpha = 0.1


def import_factorize(cellLine, geneList):
    """ which data and which genes? """
    if cellLine == 'Melanoma':
        data = ImportMelanoma()
        m = data[geneList].loc[geneList[:-1]]
    else:
        data, g = importLINCS(cellLine)
        colnames = g.copy()
        colnames.append('control')
        dat = pd.DataFrame(data, index=g, columns=colnames)
        m = dat[geneList].loc[geneList[:-1]]

    return factorizeEstimate(m.to_numpy())


# plot ODE for these genes
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


def network_odesol(ax, cellLine: str, list_gene: list):
    """ Given the list of genes and the cell line of interest plots the ode response and network. """

    # run factorization
    w, eta = import_factorize(cellLine, list_gene)

    # plot network
    Network(pd.DataFrame(w, index=list_gene[:-1], columns=list_gene[:-1]), ax[0])
    ax[0].axis('off')
    ax[0].set_title("gene-gene network " + cellLine)

    # plot ode solution over time
    ts = np.arange(0.0, 48, 4)
    sol = solver(w, eta[0], ts)

    ax[1].plot(ts, sol)
    ax[1].legend(list_gene)
    ax[1].set_ylabel("expr. level " + cellLine)
    ax[1].set_ylim((0.0, 2750))
    ax[1].set_xlabel("time")


def makeFigure():

    ax, f = getSetup((10, 10), (4, 4))
    melanoma_gene_list = ['FOSL1', 'FOXC2', 'JUN', 'JUNB', 'neg']
    gene_list1 = ['FOS', 'FOSL1', 'FOXJ3', 'FOXO3', 'FOXO4', 'JUN', 'control']
    gene_list2 = ['NFKB2', 'NFKBIA', 'NFKBIB', 'NFKBIE', 'STAT1', 'STAT3', 'STAT5B', 'control']

    network_odesol(ax[0:2], 'A375', gene_list1)
    network_odesol(ax[2:4], 'A375', gene_list2)

    network_odesol(ax[4:6], 'A549', gene_list1)
    network_odesol(ax[6:8], 'A549', gene_list2)

    network_odesol(ax[8:10], 'HA1E', gene_list1)
    network_odesol(ax[10:12], 'HA1E', gene_list2)

    network_odesol(ax[12:14], 'HT29', gene_list1)
    network_odesol(ax[14:16], 'HT29', gene_list2)

    return f
