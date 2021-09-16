"""
This creates Figure 3
"""
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, remove, normalize
from ..grndb_network import load_w_GRNdb, Network_GRNdb

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((50, 25), (1, 2))

    selected_genes(ax)
    # Add subplot labels
    subplotLabel(ax, fntsize=50)

    return f

def selected_genes(ax):
    """
    Plot the network of a set of genes with high confidence in GRNbd network, and the same genes in melanoma dataset.
    """

    # Load GRNdb W matrix
    w_GRNdb = load_w_GRNdb()
    # select genes with high confidence
    filter_row = (w_GRNdb == 2).any(axis=1)
    # filter the w matrix based on the filter
    w_2 = w_GRNdb.loc[filter_row, filter_row]

    Network_GRNdb(w_2, ax[0])
    ax[0].set_title("w Network Graph GRNdb dataset")

    # Load melanoma W matrix
    w = load_w()
    w = normalize(w)
    w = remove(w)
    # select genes and create a new dataframe for it
    w_mel = w.loc[w_2.index, list(w_2.columns)]

    Network(w_mel, ax[1])
    ax[1].set_title("w Network Graph Melanoma dataset")
