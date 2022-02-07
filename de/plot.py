""" Contains functions for graphing w matrix characteristics. """

from math import ceil
import matplotlib.pyplot as plt
import numpy as np
def plot_norm_graph(cell_lines):

    ncols = 3
    nplots = len(cell_lines) * (len(cell_lines) - 1) / 2
    nrows = ceil(nplots / ncols)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 6 * nrows), 
    squeeze=0, sharex=False, sharey=True)
    axes = np.array(axes)

    labels = []
    norms = []

    for i in range(0, len(cell_lines) - 1):
        for j in range(i + 1, len(cell_lines)):
            cellLine1 = cell_lines[i]
            cellLine2 = cell_lines[j]
            label_list = [cellLine1, cellLine2, "Difference"]
            labels.append(label_list)
            norms.append(MatrixSubtraction(cellLine1, cellLine2)[0:3])

    for i, ax in enumerate(axes.reshape(-1)):
        x = [1, 2, 3]  # number of bars
        values = norms[i]
        xlabels = labels[i]
        ax.set_title(xlabels[0] + ' vs. ' + xlabels[1] + ' (Norms)')
        ax.bar(x, height=values)
        ax.axes.set_xticks(x)
        ax.axes.set_xticklabels(xlabels)
        for i, v in enumerate(values):
            ax.text(i + 0.88, v + 0.2, "%.3f" % v, va='center')

    plt.savefig('norm_graphs.png')


def plot_impute_graph(cellLine):
    """ Tests the cross val function that creates the train and test data. """

    data, _ = importLINCS(cellLine)
    train_Y, test_Y = split_data(data)
    full_Y = impute(train_Y)

    missing = np.isnan(train_Y)

    keep_full = full_Y[missing]
    keep_test = test_Y[missing]

    print(keep_full)
    print(keep_test)
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(7.5 * 1, 6 * 1)) 
    for i, ax in enumerate(axes.reshape(-1)):
        ax.set_title('A375 Cross-Validation')
        ax.scatter(keep_full, keep_test)
        ax.set_xlabel('Predicted Data')
        ax.set_ylabel('Test Data')
        # trendline
        z = np.polyfit(keep_full, keep_test, 1)
        p = np.poly1d(z)
        ax.plot(keep_full, p(keep_full), color='red')

    plt.savefig('A375_imputation.png')
    print(np.ma.corrcoef(np.ma.masked_invalid(keep_full), np.ma.masked_invalid(keep_test))[1, 0])
