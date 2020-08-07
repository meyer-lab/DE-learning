"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, add_nodes, add_edges, pagerank, adjustment


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    G = nx.DiGraph()
    w = load_w()
    add_nodes(G, w)
    G, threshold, w_max = add_edges(G, w)
    labels = nx.get_node_attributes(G, "gene")
    pos = nx.circular_layout(G)
    ax[0].set_title("w Network Graph")
    
    pagerank(G, pos)
    edges, colors, weights = adjustment(G, threshold, w_max)
    
    nx.draw_networkx(G, pos, labels=labels, edges=edges, edge_color=colors, width=weights, font_size=8)

    # Add subplot labels
    subplotLabel(ax)
                               
    return f
