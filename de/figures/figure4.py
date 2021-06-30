"""
This creates Figure 4
"""

from math import sqrt
from .figureCommon import subplotLabel, getSetup
from ..graph import add_nodes, add_edges, remove_isolates, set_edges, set_nodes, set_labels
from .figure3 import makeFigure
import networkx as nx

above = ["JUN", "BRD2", "STK11", "PKN2", "NFAT5", "KMT2D", "ADCK3", "FOSL1", "CSK", "BRD8", "CBFB", "TADA2B", "DSTYK", "JUNB", "LATS2", "FEZF2", "MITF", "RUNX3", "SUV420H1", "SOX10", "DOT1L", "PRKRIR"] 
below = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A", "EGFR", "EP300", "SRF", "PRKAA1", "GATA4", "MYBL1", "MTF1"]

def Networkedit(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, then draw the corresponding Networkx graph.
    """
    G = nx.DiGraph()
    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)
    pos = nx.nx_pydot.pydot_layout(G, prog="fdp",scale=100.0, k=100.0)
    # draw the nodes, edges and labels
    set_nodes(G, pos, ax)
    set_edges(G, w_abs, w_max, pos, ax)
    labels = nx.get_node_attributes(G, "gene")
    set_labels(G, pos, ax)
    #set title for graph
    ax[0].set_title("w Network Graph (downstream)")
    # determine node color 
    color_map = []
    for nodes in G: 
        if labels in above :
            color_map.append('red')
        elif labels in below:
            color_map.append('blue')
    return G