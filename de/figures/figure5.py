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
    k = [k, M, n, N][0]
    pmf_edges = PMF([k, M, n, N])
    cdf_edges = CDF([k, M, n, N])

    # Make distribution plots
    ax[2].plot(k, pmf_edges, 'm-.')
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("hypergeom PMF")
    ax[3].plot(k, cdf_edges, 'y-.')
    ax[3].set_xlabel("k")
    ax[3].set_ylabel("hypergeom CDF")

    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f
