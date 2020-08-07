"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, add_nodes, add_edges


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    G = nx.DiGraph()
    w = load_w()
    add_nodes(G, w)
    G, threshold, w_max = add_edges(G, w)
    ax[0].set_title("w Network Graph")
    labels = nx.get_node_attributes(G, "gene")
    pos = nx.spring_layout(G, seed=0)
    
    #adjust edges color and thickness based on interaction type and weights
    edges = G.edges()
    colors = [G[u][v]["color"] for u,v in edges]
    weights = [np.exp((abs(G[u][v]['weight']) - threshold) / (w_max - threshold)) for u,v in edges]
    
    nx.draw_networkx(G, pos, labels=labels, edges=edges, edge_color=colors, width=weights, node_size=300, font_size=8)

    # Add subplot labels
    subplotLabel(ax)
                               
    return f
