from .factorization import cellLineFactorization, MatrixSubtraction
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def calcNorms(cellLine1, cellLine2):
    
    _, difference_norm, w1, w2 = MatrixSubtraction(cellLine1, cellLine2)

    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    np.random.shuffle(w1)
    np.random.shuffle(w2)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    return norm1, norm2, difference_norm


def plot_norm_graph(cell_lines):
    
    """
    Plot bar graphs for combinations of cell lines given a list of cell line names.
    """

    ncols = 3
    nplots = len(cell_lines)*(len(cell_lines)-1)/2
    nrows = ceil(nplots/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5*ncols, 6*nrows), squeeze=0, sharex=False, sharey=True)
    axes = np.array(axes)

    labels = []
    norms = []

    for i in range(0,len(cell_lines)-1):
        for j in range(i+1,len(cell_lines)):
            cellLine1 = cell_lines[i]
            cellLine2 = cell_lines[j]
            label_list = [cellLine1, cellLine2, "Difference"]
            labels.append(label_list)
            norms.append(calcNorms(cellLine1, cellLine2))


    for i, ax in enumerate(axes.reshape(-1)):
        x = [1,2,3] # number of bars
        values = norms[i]
        xlabels = labels[i]
        ax.set_title(xlabels[0] + ' vs. ' + xlabels[1] + ' (Norms)')
        ax.bar(x, height=values)
        ax.axes.set_xticks(x)
        ax.axes.set_xticklabels(xlabels)
        for i, v in enumerate(values):
            ax.text(i+0.88, v+0.2,"%.3f" % v, va='center')

    plt.savefig('norm_graphs.png')

def plot_entries_scatterplot(cellLine1, cellLine2):
    _, _, w1, w2 = MatrixSubtraction(cellLine1, cellLine2)
    w1_edit = w1.flatten()
    w2_edit = w2.flatten()
    plt.scatter(w1_edit, w2_edit, marker = 'o')
    plt.title("Flattened Matrice Graph")
    plt.savefig('w1_w2graph.png')

def plot_corr_graphs(cell_lines):
    """Plot all combinations of two w matrices against each other (still using factorizeEstimate instead of runOptim)"""

    ncols = 3
    nplots = len(cell_lines)*(len(cell_lines)-1)/2
    nrows = ceil(nplots/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5*ncols, 6*nrows), squeeze=0, sharex=False, sharey=True)
    axes = np.array(axes)

    labels = []
    data = []

    for i in range(0,len(cell_lines)-1):
        for j in range(i+1,len(cell_lines)):
            cellLine1 = cell_lines[i]
            cellLine2 = cell_lines[j]
            label_list = [cellLine1, cellLine2]
            labels.append(label_list)
            _, _, w1, w2 = MatrixSubtraction(cellLine1, cellLine2)
            data.append([w1.flatten(), w2.flatten()])


    for i, ax in enumerate(axes.reshape(-1)):
        label_list = labels[i]
        values = data[i]
        ax.set_title(label_list[1] + ' vs. ' + label_list[0])
        ax.scatter(values[0], values[1])
        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.set_xlabel(label_list[0])
        ax.set_ylabel(label_list[1])

    plt.savefig('corr_graphs.png')

