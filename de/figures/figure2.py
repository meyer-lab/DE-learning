"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, remove_POLR2A, add_nodes, add_edges, remove_isolates, set_nodes, set_edges, set_labels

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    #create NetworkX graph
    G = nx.DiGraph()
    w = load_w()
    w = remove_POLR2A(w)
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)

    #add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)

    #draw the nodes, edges and labels
    pos = nx.spring_layout(G, k=8.0/G.number_of_nodes())
    set_nodes(G, pos)
    set_edges(G, w_abs, w_max, pos)
    set_labels(G, pos)

    ax[0].set_title("w Network Graph")

    # Add subplot labels
    subplotLabel(ax)
                   
    return f
