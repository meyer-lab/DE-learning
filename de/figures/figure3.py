"""
This creates Figure 2: w Network Graph
"""
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
import numpy as np
import networkx as nx
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, normalize, remove, bar_graph


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
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
    
    full, pre, rand = cluster_dist()
    # TODO: find the equivalent values of the w matrix for full, pre, and random indexes, and use ax[4].hist to plot them 
    # ax[4].hist()
    ax[5].axis('off')
    return f

def cluster_dist(G):
    w = load_w()
    fullR = w[26, 1, 69, 48, 42, 31, 0, 18, 9, 2, 4, 71, 13, 27, 32, 17, 39, 61, 70, 64, 12, 54]
    preR = w[36, 35, 46, 28, 15, 16, 68, 52, 20, 41, 40, 16, 46]
    full = []
    pre = []
    rand = []

    for _ in range(50):
        full.append(np.random.choice(fullR, 2))
        pre.append(np.random.choice(preR, 2))
        rand.append(np.concatenate([np.random.choice(fullR,1), np.random.choice(preR, 1)]))
    full_dij = single_source_dijkstra(G, source=str(full[0]), target=str(full[1]), weight=True)
    pre_dij = single_source_dijkstra(G, source=pre[0], target=pre[1], weight=True)
    rand_dij = single_source_dijkstra(G, source=rand[0], target=rand[1], weight=True)
    return full_dij, pre_dij, rand_dij
