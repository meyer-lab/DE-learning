"""
This creates Figure 3
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..importData import prepData

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((12, 8), (1, 2))
    data = prepData()
    figureMaker(ax, data)

    # Add subplot labels
    subplotLabel(ax)

    return f

def figureMaker(ax, data):

    # PC 1
    labels = ["MITF", "SOX10", "NSD1", "SRF"]
    mitf = []
    sox10 = []
    nsd1 = []
    srf = []

    width = 0.2
    x = np.arange(len(labels))

    for j in labels:
        mitf.append(data.loc[j, "MITF"])
        sox10.append(data.loc[j, "SOX10"])
        nsd1.append(data.loc[j, "NSD1"])
        srf.append(data.loc[j, "SRF"])

    i = 0
    ax[i].bar(x - 3 * width / 2, mitf, width, label='MITF Knockout')
    ax[i].bar(x - width/2, sox10, width, label='SOX10 Knockout')
    ax[i].bar(x + width/2, nsd1, width, label='NSD1 Knockout')
    ax[i].bar(x + 3 * width / 2, srf, width, label='SRF Knockout')
    ax[i].set_ylabel('RNAseq Gene Expression Measurement (reads per million)')
    ax[i].set_title('PC1')
    ax[i].set_xlabel('Gene')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].legend(loc=2)

    # PC 3
    labels = ["POLR2A", "SOX10", "STK11", "MITF"]
    polr2a = []
    sox10 = []
    stk11 = []
    mitf = []

    for j in labels:
        polr2a.append(data.loc[j, "POLR2A"])
        sox10.append(data.loc[j, "SOX10"])
        stk11.append(data.loc[j, "STK11"])
        mitf.append(data.loc[j, "MITF"])

    i = 1
    ax[i].bar(x - 3 * width / 2, polr2a, width, label='POLR2A Knockout')
    ax[i].bar(x - width/2, sox10, width, label='SOX10 Knockout')
    ax[i].bar(x + width/2, stk11, width, label='STK11 Knockout')
    ax[i].bar(x + 3 * width / 2, mitf, width, label='MITF Knockout')
    ax[i].set_ylabel('RNAseq Gene Expression Measurement (reads per million)')
    ax[i].set_title('PC3')
    ax[i].set_xlabel('Gene')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].legend(loc=2)
