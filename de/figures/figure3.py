"""
This creates Figure 2: w Network Graph
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, remove, normalize, bar_graph
from ..grndb_network import load_w_GRNdb, Network_GRNdb

def makeFigure():
    """ Get a list of the axis objects and create a figure.
    :output f: Figure 2 containing network diagram, path length bar graph, and upstream and downstream bar graphs
    :type f: Figure
    """
    # Get list of axis objects
    ax, f = getSetup((150, 100), (2, 3))
    # load w for the Melanoma dataset from Torre paper
    w = load_w()
    w = normalize(w)
    w = remove(w)
    # Plot downstream graph
    Network(w, w_abs, w_max, ax[0])
    # set title for the graph
    ax[0].set_title("w Network Graph (downstream)")
    # create downstream bar graph
    bar_graph(w, "green", ax[1], "downstream")
    # set title for the graph
    ax[1].set_title("Bar Graph (downstream)")
    # Plot upstream graph
    w_trans = np.transpose(w)
    w_abs = np.absolute(w_trans.to_numpy())
    w_max = np.max(w_abs)

    # create upstream bar graph
    bar_graph(w_trans, "orange", ax[2], "upstream")
    # set title for the graph
    ax[2].set_title("Bar Graph (upstream)")

    # Plot Mia's network (GRNdb) 
    w_GRNdb = load_w_GRNdb()
    Network_GRNdb(w_GRNdb, ax[3])
    ax[3].set_title("w Network Graph - GRNdb")

    # Plot nothing in the place of ax[5]
    ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f
