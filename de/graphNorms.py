from .factorization import cellLineComparision, MatrixSubtraction
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def plot_norm_graph(diff_norms, cell_line_norms, labels):
    """
    Plot bar graphs for sets of cell lines given a matrix of difference norms, a list of individual norms, and a list of cell line names.
    """

    ncols = 3
    nplots = len(cell_line_norms)*(len(cell_line_norms)-1)/2
    nrows = ceil(nplots/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5*ncols, 6*nrows), squeeze=0, sharex=False, sharey=True)
    axes = np.array(axes)

    # to do: 

    for i, ax in enumerate(axes.reshape(-1)):
        x = [1,2,3] # number of bars
        values = [1,2,2]
        ax.set_title(f'Subplot: {i}')
        ax.bar(x, height=values)
        #ax.axes.xaxis.set_visible(False)
        labels=['a','b','c']
        ax.axes.set_xticks(x)
        ax.axes.set_xticklabels(labels)

    plt.savefig('norm_graphs.png')
    
    
# test values:
a = np.matrix('1 2 3 4; 3 4 5 6; 4 5 6 7; 1 2 3 4')
norms = [2, 3, 2, 2.5]
labels = ['a', 'b', 'c', 'd']

plot_norm_graph(a, norms, labels)
