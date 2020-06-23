"""
This creates Figure 3
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 7. """
    ax, f = getSetup((7, 6), (3, 3))

    subplotLabel(ax)

    return f
