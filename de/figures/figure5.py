"""
This creates Figure 3
"""
import matplotlib as plt
from .figureCommon import subplotLabel, getSetup
from .figure3 import cluster_dist
from ..graph import histogram


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 8), (3, 2))
    w_full = cluster_dist()[0]
    w_pre = cluster_dist()[1]
    w_rand = cluster_dist()[2]
    histogram(w_full, w_pre, w_rand)
    #ax[0].set_title("Distance Distribution")
    # Add subplot labels
    subplotLabel(ax)
    return f
