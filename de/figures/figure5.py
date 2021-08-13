"""This creates hypergeometric distribution graphs for comparing our w Network Graph with the GRNdb Network Graph."""
import matplotlib.pyplot as plt
import numpy as np
from ..hypergeom import setvars, PMF, CDF
from ..graph import load_w, normalize, remove, Network
from ..grndb_network import load_w_GRNdb, Network_GRNdb
from .figureCommon import subplotLabel, getSetup

def makeFigure():
    """
    Get a list of the axis objects and create the figure.
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
    ax[0].set_title("w Network Graph (downstream)")
    
    # Plot GRNdb network
    w_GRNdb = load_w_GRNdb()
    G_GRNdb = Network_GRNdb(w_GRNdb, ax[1])
    ax[1].set_title("w Network Graph - GRNdb")

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
    ax[2].vlines([k, M, n, N][0], 0, 0.12, linestyles='dashed', linewidth=6, label=k_label)
    ax[2].legend()
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("hypergeom PMF")
    ax[2].set_title("Probability Mass Function")

    # Plot CDF
    ax[3].plot([xx, cdf_edges][0], [xx, cdf_edges][1], 'g-.', linewidth=6, label=curve_label)
    ax[3].vlines([k, M, n, N][0], 0, 1, linestyles='dashed', linewidth=6, label=k_label)
    ax[3].legend()
    ax[3].set_xlabel("k")
    ax[3].set_ylabel("hypergeom CDF")
    ax[3].set_title("Cumulative Distribution Function")

    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f
