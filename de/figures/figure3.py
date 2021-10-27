"""
This creates Figure 2: w Network Graph
"""
from de.factorization import SparseFactorization
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, remove, normalize, bar_graph
from ..grndb_network import load_w_GRNdb, Network_GRNdb


def makeFigure():
    """ Get a list of the axis objects and create a figure.
    :output f: Figure 2 containing network diagram, path length bar graph, and upstream and downstream bar graphs
    :type f: Figure
    """
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 4))
    # load w for the Melanoma dataset from Torre paper
    w = load_w()
    w = normalize(w)
    w = remove(w)
    Network(w, ax[0])

    # Plot downstream graph
    ax[0].set_title("w Network Graph (downstream)")
    bar_graph(w, "green", ax[1], "downstream")
    ax[1].set_title("Bar Graph (downstream)")

    # Plot upstream graph
    bar_graph(w.T, "orange", ax[2], "upstream")
    ax[2].set_title("Bar Graph (upstream)")

    # Plot Mia's network (GRNdb)
    w_GRNdb = load_w_GRNdb()
    Network_GRNdb(w_GRNdb, ax[3])
    ax[3].set_title("w Network Graph - GRNdb")

    # Plot network with sparsity
    w_sparse = SparseFactorization(w)
    Network(w_sparse, ax[4])
    ax[4].set_title("w Network Graph - Sparsity Added")

    # Add subplot labels
    subplotLabel(ax)
    return f
