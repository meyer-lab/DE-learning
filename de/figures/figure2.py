"""
This creates Figure 2: w Network Graph
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..graph import G, load_w, remove_POLR2A


def makeFigure(pagerank_threshold = None):
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((12, 12), (1, 2))
    
    #load w
    w = load_w()
    w = remove_POLR2A(w)
    
    #Plot downstream graph
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
    
    #create NetworkX graph
    G_downstream = G(w, w_abs, w_max)
    
    #set title for the graph
    ax[0].set_title("w Network Graph (downstream)")
    
    #Plot upstream graph
    w_trans = np.transpose(w)
    w_abs_trans = np.absolute(w_trans.to_numpy())
    w_max = np.max(w_abs_trans)
    
    #create NetworkX graph
    G_upstream = G(w_trans, w_abs_trans, w_max)
    
    #set title for the graph
    ax[1].set_title("w Network Graph (upstream)")
    
    # Add subplot labels
    subplotLabel(ax)
                   
    return f
