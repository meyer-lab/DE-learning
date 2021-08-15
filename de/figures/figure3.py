"""
This creates Figure 2: w Network Graph
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, normalize, remove, bar_graph


def makeFigure():
    """ Get a list of the axis objects and create a figure.

    :output f: Figure 2 containing the network diargam, path bar graph, and upstream and downstream bar graphs
    :type f: Figure
    """
    # Get list of axis objects
    ax, f = getSetup((100, 100), (2, 2))
    # load w
    w = load_w()
    w = normalize(w)
    w = remove(w)
    # Plot downstream graph
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
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
    bar_graph(w_trans, "orange", ax[3], "upstream")
    # set title for the graph
    ax[3].set_title("Bar Graph (upstream)")
    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f
