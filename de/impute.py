""" This file includes the functions for cross-validation based on data imputation. """
import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns
import itertools
from .factorization import factorizeEstimate
from .importData import importLINCS, ImportMelanoma

def split_data(X, n=10):
    """ Prepare the test and train data. """
    row = np.random.choice(X.shape[0], 1, replace=False)
    col = np.random.choice(X.shape[1], n, replace=False)

    train_X = np.copy(X)
    test_X = np.full_like(X, np.nan)
    train_X[row, col] = np.nan
    test_X[row, col] = X[row, col]
    assert np.sum(np.isnan(train_X)) == n
    assert np.sum(np.isfinite(test_X)) == n
    return train_X, test_X


def impute(data):
    """ Impute by repeated fitting. """
    # Fit
    _, _, data = factorizeEstimate(data, maxiter=50, returnData=True)
    return data[0]


def repeatImputation(data, linear=False, numIter=20):
    """ Repeat imputation and calculate the average of cost for 20 iterations. """
    coefs = []
    for _ in range(numIter):
        train_X, test_X = split_data(data)
        full_X = impute(train_X)
        corr_coef = ma.corrcoef(ma.masked_invalid(full_X.flatten()), ma.masked_invalid(test_X.flatten()))
        coefs.append(corr_coef[0][1])
    print(f"average corr coef: {sum(coefs)/len(coefs)}")
    return coefs

def calc_imputation():
    """ plot correlation coefficients as a boxplot. """
    data_list = [importLINCS("A375")[0], importLINCS("A549")[0], importLINCS("HA1E")[0], importLINCS("HT29")[0], importLINCS("MCF7")[0], importLINCS("PC3")[0], ImportMelanoma()]

    # run the repeated imputation for nonlinear and linear model
    nonlinear_coeffs = []
    linear_coeffs = []
    for data in data_list:
        linear_coeffs.append(repeatImputation(data, linear=True))
        nonlinear_coeffs.append(repeatImputation(data))

    return linear_coeffs, nonlinear_coeffs

def plot_imputation(ax):
    """ plots the boxplot of correlation coefficient for 6 LINCS cell lines and Melanoma in linear and nonlinear case. """
    linear, nonlinear = calc_imputation()

    n = len(linear[0])
    labels = 2 * [["A375"] * n, ["A549"] * n, ["HA1E"] * n, ["HT29"] * n, ["MCF7"] * n, ["PC3"] * n, ["Mel"] * n]
    hue = [["linear"] * 5 * n, ["nonlinear"] * 5 * n]
    df = pd.DataFrame({'correlation coef.': list(itertools.chain(linear + nonlinear)), 'cellLines': labels, 'model': hue})
    sns.boxplot(x='cellLines', y='correlation coef', hue='model', data=df, ax=ax, split=True, jitter=0.2, palette=sns.color_palette('Paired'))
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[0:2], labels[0:2],
                       loc='upper left',
                       fontsize='large',
                       handletextpad=0.5)