"""
This creates Figure 2: w Network Graph
"""
import numpy as np
import random
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, normalize, remove, bar_graph


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((100, 150), (3, 2))
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

    full, pre, rand = cluster_dist()
    # TODO: find the equivalent values of the w matrix for full, pre, and random indexes, and use ax[4].hist to plot them.
    # ax[4].hist()
    ax[5].axis('off')
    return f

def cluster_dist():
    fullR = ["JUN", "BRD2", "PKN2", "NFAT5"]
    preR = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A"]
    full = []
    pre = []
    rand = []

    for _ in range(50):
        full.append(random.sample(fullR, 2))
        pre.append(random.sample(preR, 2))
        rand.append([random.sample(fullR, 1)[0], random.sample(preR, 1)[0]])

    return full, pre, rand


