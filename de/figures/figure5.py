"""This creates hypergeometric distribution graphs for comparing our w Network Graph with the GRNdb Network Graph."""
import matplotlib.pyplot as plt
import numpy as np
from ..hypergeom import setvars, PMF, CDF
from ..graph import load_w, normalize, remove, Network
from ..grndb_network import load_w_GRNdb, Network_GRNdb
from .figureCommon import subplotLabel, getSetup

def makeFigure():
    """
    Gets a list of axes objects and creates the figure.
    
    :output f: Figure 5 containing network diagrams for the w matrix and the GRNdb dataset, along with 
    the probability mass and cumulative distribution function plots for the significance of their overlapping edges.
    :type f: Figure
    """    
    # Get list of axis objects
    ax, f = getSetup((100, 100), (2,2))
    # load w
    w = load_w()
    w = normalize(w)
    w = remove(w)
    # Plot downstream graph
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
    G = Network(w, w_abs, w_max, ax[0])
    # set title for the graph
    ax[0].set_title("w Network Graph (downstream)", fontsize='large')
    
    # Plot GRNdb network
    w_GRNdb = load_w_GRNdb()
    G_GRNdb = Network_GRNdb(w_GRNdb, ax[1])
    ax[1].set_title("w Network Graph - GRNdb", fontsize='large')

    # Get hypergeometric dist/ variables
    [k, M, n, N] = setvars(G, G_GRNdb)
    kstr = str([k, M, n, N][0])
    Mstr = str([k, M, n, N][1])
    nstr = str([k, M, n, N][2])
    Nstr = str([k, M, n, N][3])
    [x, pmf_edges] = PMF([k, M, n, N])
    [xx, cdf_edges] = CDF([k, M, n, N])

    # Set labels
    curve_label = 'M = ' + Mstr + ', n = ' + nstr + ',  N = ' + Nstr
    k_label = 'k = ' + kstr

    # Plot PMF
    ax[2].plot([x, pmf_edges][0], [x, pmf_edges][1], 'm-.', linewidth=6, label=curve_label)
    ax[2].vlines([k, M, n, N][0], 0, 0.2, linestyles='dashed', linewidth=6, label=k_label)
    ax[2].legend(prop={"size":70})
    ax[2].set_xlabel("k", fontsize='large')
    ax[2].set_ylabel("hypergeom PMF", fontsize='large')
    ax[2].tick_params(axis='both', which='major', labelsize=70)
    ax[2].set_title("Probability Mass Function", fontsize='large')

    # Plot CDF
    ax[3].plot([xx, cdf_edges][0], [xx, cdf_edges][1], 'g-.', linewidth=6, label=curve_label)
    ax[3].vlines([k, M, n, N][0], 0, 1, linestyles='dashed', linewidth=6, label=k_label)
    ax[3].legend(prop={"size":70})
    ax[3].set_xlabel("k", fontsize='large')
    ax[3].set_ylabel("hypergeom CDF", fontsize='large')
    ax[3].tick_params(axis='both', which='major', labelsize=70)
    ax[3].set_title("Cumulative Distribution Function", fontsize='large')

    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f
