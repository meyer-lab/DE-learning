"""
This creates Figure 3
"""
from .figureCommon import subplotLabel, getSetup


def Network(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, then draw the corresponding Networkx graph.
    """
    G = nx.DiGraph()
    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)

    pos = nx.nx_pydot.pydot_layout(G, prog="fdp")

    # draw the nodes, edges and labels
    set_nodes(G, pos, ax)
    set_edges(G, w_abs, w_max, pos, ax)
    set_labels(G, pos, ax)
    # determines node color based on 
    color_map = []
    for node in G: 
        if node > x=y:
            color_map.append('red')
        else:
            color_map.append('blue')

    return G