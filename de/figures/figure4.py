"""
This creates Figure 3
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    return f
