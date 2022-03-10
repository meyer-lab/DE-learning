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

def makeFigure():

    ax, f = getSetup((6, 12), (5, 2))
    melanoma_gene_list = ['FOSL1', 'FOXC2', 'JUN', 'JUNB', 'neg']
    A375_gene_list = ['FOS', 'FOSL1', 'FOXJ3', 'FOXO3', 'FOXO4', 'JUN', 'control']

    w_mel, eta_mel = import_factorize('Melanoma', melanoma_gene_list)
    w_375, eta_375 = import_factorize('A375', A375_gene_list)
    w_549, eta_549 = import_factorize('A549', A375_gene_list)
    w_ha1e, eta_ha1e = import_factorize('HA1E', A375_gene_list)
    w_ht29, eta_ht29 = import_factorize('HT29', A375_gene_list)

    # plot network
    Network(pd.DataFrame(w_mel, index=melanoma_gene_list[:-1], columns=melanoma_gene_list[:-1]), ax[0])
    Network(pd.DataFrame(w_375, index=A375_gene_list[:-1], columns=A375_gene_list[:-1]), ax[2])
    Network(pd.DataFrame(w_549, index=A375_gene_list[:-1], columns=A375_gene_list[:-1]), ax[4])
    Network(pd.DataFrame(w_ha1e, index=A375_gene_list[:-1], columns=A375_gene_list[:-1]), ax[6])
    Network(pd.DataFrame(w_ht29, index=A375_gene_list[:-1], columns=A375_gene_list[:-1]), ax[8])

    for i in [0, 2, 4, 6, 8]:
        ax[i].axis('off')
    ax[0].set_title("gene-gene network Melanoma")
    ax[2].set_title("gene-gene network A375")
    ax[4].set_title("gene-gene network A549")
    ax[6].set_title("gene-gene network HA1E")
    ax[8].set_title("gene-gene network HT29")

    # plot over time
    ts = np.arange(0.0, 48, 4)
    # melanoma
    sol = solver(w_mel, eta_mel[0], ts)
    # A375
    sol375 = solver(w_375, eta_375[0], ts)
    # A549
    sol549 = solver(w_549, eta_549[0], ts)
    # HA1E
    solha1e = solver(w_ha1e, eta_549[0], ts)
    # HT29
    solht29 = solver(w_ht29, eta_ht29[0], ts)

    ax[1].plot(ts, sol)
    ax[1].legend(melanoma_gene_list)
    ax[1].set_ylabel("expr. level Melanoma")

    ax[3].plot(ts, sol375)
    ax[3].legend(A375_gene_list)
    ax[3].set_ylabel("expr. level A375")

    ax[5].plot(ts, sol549)
    ax[5].legend(A375_gene_list)
    ax[5].set_ylabel("expr. level A549")

    ax[7].plot(ts, solha1e)
    ax[7].legend(A375_gene_list)
    ax[7].set_ylabel("expr. level HA1E")

    ax[9].plot(ts, solht29)
    ax[9].legend(A375_gene_list)
    ax[9].set_ylabel("expr. level HT29")

    for i in [1, 3, 5, 7, 9]:
        ax[i].set_ylim((0.0, 2750))
        ax[i].set_xlabel("time")

    return f
