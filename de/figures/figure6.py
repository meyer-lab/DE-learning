""" This file plots the imputation results for all LINCS and melanoma data that we have. """
import pandas as pd
import seaborn as sns
import itertools

from ..importData import importLINCS, ImportMelanoma
from ..impute import repeatImputation
from .common import subplotLabel, getSetup


def makeFigure():
    """
    Plot figure 6, plot boxplot for correlation coefficients of impution results comapring linear and nonlinear model.
    """
    # Get list of axis objects
    ax, f = getSetup((7, 5), (1, 1))
    linear, nonlinear = plot_imputation()

    n = len(linear[0])
    labels = 2 * [["A375"] * n, ["A549"] * n, ["HA1E"] * n, ["HT29"] * n, ["MCF7"] * n, ["PC3"] * n, ["Mel"] * n]
    hue = [["linear"] * 5 * n, ["nonlinear"] * 5 * n]
    df = pd.DataFrame({'correlation coef.': list(itertools.chain(linear + nonlinear)), 'cellLines': labels, 'model': hue})
    sns.boxplot(x='cellLines', y='correlation coef', hue='model', data=df, ax=ax[0], split=True, jitter=0.2, palette=sns.color_palette('Paired'))
    handles, labels = ax[0].get_legend_handles_labels()
    lgd = ax[0].legend(handles[0:2], labels[0:2],
                       loc='upper left',
                       fontsize='large',
                       handletextpad=0.5)

    # Add subplot labels
    subplotLabel(ax)
    return f


def plot_imputation():
    """ plot correlation coefficients as a boxplot. """
    data_list = [importLINCS("A375")[0], importLINCS("A549")[0], importLINCS("HA1E")[0], importLINCS("HT29")[0], importLINCS("MCF7")[0], importLINCS("PC3")[0], ImportMelanoma()]

    # run the repeated imputation for nonlinear and linear model
    nonlinear_coeffs = []
    linear_coeffs = []
    for data in data_list:
        linear_coeffs.append(repeatImputation(data, linear=True))
        nonlinear_coeffs.append(repeatImputation(data))

    return linear_coeffs, nonlinear_coeffs
