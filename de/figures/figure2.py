"""
This creates Figure 2: w Network Graph
"""
import networkx as nx
from .figureCommon import subplotLabel, getSetup
from ..graph import load_w, add_nodes, add_edges


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    G = nx.DiGraph()
    w = load_w()
    add_nodes(G, w)
    add_edges(G, w)
    ax[0].set_title("w Network Graph")
    labels = nx.get_node_attributes(G, 'gene')
    nx.draw_networkx(G,labels=labels, node_size=200, font_size=8)

    # Add subplot labels
    subplotLabel(ax)

    return f
