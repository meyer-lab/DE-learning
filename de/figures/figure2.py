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
    
    #create NetworkX graph
    G = nx.DiGraph()
    w = load_w()
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
    
    #add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    threshold(G)
    
    #draw the nodes, edges and labels
    pos = nx.spring_layout(G)
    set_nodes(G, pos)
    set_edges(G, w_abs, w_max, pos)
    set_labels(G, pos)

    ax[0].set_title("w Network Graph")
    plt.show()
    
    # Add subplot labels
    subplotLabel(ax)
                               
    return f
