"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, add_nodes, add_edges, threshold, set_nodes, set_edges, set_labels
import matplotlib.pyplot as plt
import numpy as np

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    G = nx.DiGraph()
    w = load_w()
    w_new = abs(w.to_numpy())
    w_max = np.max(w_new)

    add_nodes(G, w, w_new)
    add_edges(G, w, w_new)
    threshold(G)
    
    pos = nx.spring_layout(G)
    set_nodes(G, pos)
    set_edges(G, w_new, w_max, pos)
    set_labels(G, pos)

    ax[0].set_title("w Network Graph")
    plt.show()
    
    # Add subplot labels
    subplotLabel(ax)
                               
    return f
