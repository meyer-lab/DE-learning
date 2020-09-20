"""
This creates Figure 3
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 3))

    time_sols = np.loadtxt("time_sols.csv", delimiter=',')
    figureMaker(ax, time_sols)

    # Add subplot labels
    subplotLabel(ax)

    return f

def figureMaker(ax, time_sols):
    t = list(range(1, 9990, 10))

    for i in (0, 3, 6):
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("MITF")
        ax[i].set_yticks(range(0, 3000, 500))
        ax[i].set_ylim(bottom=0, top=3000)
        ax[i].hlines(151.3545966557261, 0, 10000, colors='r', linestyles="dashed")
        ax[i].hlines(75.4331554601265, 0, 10000, colors='r', linestyles="dashed")
        ax[i].hlines(954.8606605977672, 0, 10000, colors='r', linestyles="dashed")
    for i in (1, 4, 7):
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("YBX1")
        ax[i].set_yticks(range(0, 3000, 500))
        ax[i].set_ylim(bottom=0, top=3000)
        ax[i].hlines(986.0857413773354, 0, 10000, colors='r', linestyles="dashed")
        ax[i].hlines(175.08803202992868, 0, 10000, colors='r', linestyles="dashed")
        ax[i].hlines(1679.7608608181388, 0, 10000, colors='r', linestyles="dashed")
    for i in (2, 5, 8):
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("SOX10")
        ax[i].set_yticks(range(0, 3000, 500))
        ax[i].set_ylim(bottom=0, top=3000)
        ax[i].hlines(1124.4605150447003, 0, 10000, colors='r', linestyles="dashed")
        ax[i].hlines(235.1749762869939, 0, 10000, colors='r', linestyles="dashed")

    i = 0
    ax[i].plot(t, time_sols[39, 0:999])
    i = 1
    ax[i].plot(t, time_sols[78, 0:999])
    i = 2
    ax[i].plot(t, time_sols[64, 0:999])
    i = 3
    ax[i].plot(t, time_sols[39, (17 * 1000):(17 * 1000) + 999])
    i = 4
    ax[i].plot(t, time_sols[78, (17 * 1000):(17 * 1000) + 999])
    i = 5
    ax[i].plot(t, time_sols[64, (17 * 1000):(17 * 1000) + 999])
    i = 6
    ax[i].plot(t, time_sols[39, (69 * 1000):(69 * 1000) + 999])
    i = 7
    ax[i].plot(t, time_sols[78, (69 * 1000):(69 * 1000) + 999])
    i = 8
    ax[i].plot(t, time_sols[64, (69 * 1000):(69 * 1000) + 999])
