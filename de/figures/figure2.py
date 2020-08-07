"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, add_nodes, add_edges, pagerank, adjustment
import matplotlib.pyplot as plt

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    G = nx.DiGraph()
    w = load_w()
    add_nodes(G, w)
    G, threshold, w_max = add_edges(G, w)
    labels = nx.get_node_attributes(G, "gene")
    pos = nx.spring_layout(G)
    ax[0].set_title("w Network Graph")
    
    #adjust the size of node based on pagerank
    pagerank(G, pos)
    #adjust the thickness and color of edges based on weights and interaction type separately
    adjustment(G, threshold, w_max, pos)
    #draw the label for each node
    nx.draw_networkx_labels(G, pos, labels = labels, font_size=10)
    plt.show()
    
    # Add subplot labels
    subplotLabel(ax)
                               
    return f
