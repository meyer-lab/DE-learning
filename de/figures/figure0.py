""" Analyzing a subset of genes. """
from ..importData import ImportMelanoma, importLINCS
from ..factorization import factorizeEstimate, commonGenes
from .figureCommon import subplotLabel, getSetup
from ..graph import normalize, Network
import pandas as pd
import numpy as np
import matplotlib, matplotlib.pyplot as plt

def makeFigure():
    """ make network for selected genes. """
    ax, f = getSetup((12, 12), (3, 3))

    data1, genes1 = ImportMelanoma()
    # normalized to control

    data2, genes2 = importLINCS("A375")
    # normalize to control
    data2 /= 10.0

    index_list1, index_list2 = commonGenes(genes1, genes2)
    common_genes = list(np.array(genes1)[index_list1])

    selected_genes1 = select_genes(data1, genes1, common_genes)
    selected_genes2 = select_genes(data2, genes2, common_genes)
    # selected_genes2 = select_genes(["MITF", "SOX10", "GLI4", "ZEB2", "EP300", "FOSL1"])
    # selected_genes3 = select_genes(data1, genes1, ["MITF", "GATA4", "HOXB6", "SOX10", "DSTYK"])
    ax[0].grid(False)
    ax[3].grid(False)
    ax[6].grid(False)
    Analyze_selectedGenes(selected_genes1, ax[0:3])
    Analyze_selectedGenes(selected_genes2, ax[3:6])
    # Analyze_selectedGenes(selected_genes3, ax[6:9])
    
    subplotLabel(ax)

    return f

def select_genes(data_array, all_genes, select_genes):
    """ create a data set of MAFA, SOX10, MITF, JUN, TP73, CLK1 from the Melanoma data. """

    col = all_genes.copy()
    col.append("control")
    df_data = pd.DataFrame(data_array, columns=col, index=all_genes)


    genes_and_control = select_genes.copy()
    genes_and_control.append("Control")
    # find the column number corresponding to the select genes
    select_gene_index = [all_genes.index(a) for a in select_genes]
    select_gene_index.append(-1)
    # separate the values of the data into a new dataframe
    select_data = np.zeros((len(select_genes), len(genes_and_control)))
    for i in range(len(select_genes)):
        select_data[i, :] = data_array[select_gene_index[i], select_gene_index]

    return pd.DataFrame(select_data, columns=genes_and_control, index=select_genes)

def Analyze_selectedGenes(select_data_df, ax):
    """ Fit and plot w and data for those selected genes. """
    w, eta = factorizeEstimate(select_data_df.to_numpy())

    w_df = pd.DataFrame(w, columns=select_data_df.index, index=select_data_df.index)
    w_df = normalize(w_df)
    Network(w_df, ax[0])

    im, cbar = heatmap(select_data_df, select_data_df.index, select_data_df.columns, ax[1],
                   cmap="YlGn", cbarlabel="Gene Expression")
    # annotate_heatmap(im, valfmt="{x:.1f}")

    im, cbar = heatmap(w_df, w_df.index, w_df.columns, ax[2],
                   cmap="YlGn", cbarlabel="Gene-gene Interaction")
    # annotate_heatmap(im, valfmt="{x:.1f}")


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    ax.grid(False)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts