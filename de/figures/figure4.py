"""
This creates Figure 4
"""
from .figureCommon import subplotLabel, getSetup

above = ["JUN", "BRD2", "STK11", "PKN2", "NFAT5", "KMT2D", "ADCK3", "FOSL1", "CSK", "BRD8", "CBFB", "TADA2B", "DSTYK", "JUNB", "LATS2", "FEZF2", "MITF", "RUNX3", "SUV420H1", "SOX10", "DOT1L", "PRKRIR"] 
below = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A", "EGFR", "EP300", "SRF", "PRKAA1", "GATA4", "MYBL1", "MTF1"]

def Network(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, then draw the corresponding Networkx graph.
    """
    G = nx.DiGraph()
    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)
    pos = nx.nx_pydot.pydot_layout(G, prog="fdp", scale = 3)
    # draw the nodes, edges and labels
    set_nodes(G, pos, ax)
    set_edges(G, w_abs, w_max, pos, ax)
    set_labels(G, pos, ax)
    # determines node color 
    color_map = []
    for node in G: 
        if 'set_labels' in above :
            color_map.append('red')
        elif 'set_labels' in below:
            color_map.append('blue')
    return G