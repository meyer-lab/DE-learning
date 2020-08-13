"""
This creates Figure 2: w Network Graph
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import G, G_new, load_w, remove_POLR2A


def makeFigure(pagerank_threshold = None):
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    
    w = load_w()
    w = remove_POLR2A(w)
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
    
    #create NetworkX graph
    if pagerank_threshold:
        G_1 = G(pagerank_threshold, w, w_abs, w_max)
        G_2 = G_new(G_1, w_abs, w_max)
    else:
        G_1 = G(pagerank_threshold, w, w_abs, w_max)

    #set title for the graph
    ax[0].set_title("w Network Graph")
    
    # Add subplot labels
    subplotLabel(ax)
                   
    return f
