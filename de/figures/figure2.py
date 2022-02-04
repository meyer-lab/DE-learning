"""
This creates Figure 1: PCA plots
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from ..importData import prepData
from ..tensor import factorize, form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 3))

    # Perform PCA
    data = prepData()
    pca_object = PCA(n_components=25)
    X_r = pca_object.fit_transform(normalize(data.T))

    # Create dataframe for plotting
    KO_genes_unique = list(set(data.columns))
    df = pd.DataFrame(X_r)
    df["KO Gene"] = data.columns

    # Make subplots
    figureMaker(ax, pca_object, df, KO_genes_unique)

    # tensor factorization
    tfac, r2x, _, cellLines = factorize(num_comp=9)
    cellLine_factors = pd.DataFrame(tfac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=cellLines)

    ax[3].scatter(range(1, 10), r2x, lw=2)
    ax[3].set_title("Tensor R2X")
    ax[3].set_ylabel("R2x")
    ax[3].set_xlabel("Number of Components")
    ax[3].set_ylim((0.0, 1.0))
    g1 = sns.heatmap(cellLine_factors, cmap="PRGn", center=0, yticklabels=True, cbar=True, ax=ax[4])
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
    g1.set_title("Cell lines")

    ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    return f


def figureMaker(ax, pca_object, df, KO_genes_unique):
    """ Makes 3 panels: (A) R2X, (B) PC2 vs PC1, (C) ? """
    # Plot R2X
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylabel("Exp. Variance Ratio")
    ax[0].set_xticks(np.arange(1, 26, 2))
    ax[0].plot(list(range(1, 26)), pca_object.explained_variance_ratio_, lw=3)
    ax[0].set_title("R2X")

    # Plot PC2 vs PC1
    KO_genes = df.loc[:, "KO Gene"]
    for j, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        ax[1].scatter(df.iloc[:, 0][indx], df.iloc[:, 1][indx])
    for j, txt in enumerate(KO_genes):
        ax[1].annotate(txt, (df.iloc[j, 0], df.iloc[j, 1]))
    ax[1].set_xlabel("PC1 (" + str(round(pca_object.explained_variance_ratio_[0] * 100, 2)) + "%)")
    ax[1].set_ylabel("PC2 (" + str(round(pca_object.explained_variance_ratio_[1] * 100, 2)) + "%)")
    ax[1].set_title("PC2 vs PC1")

    # Plot PC3 vs PC1
    KO_genes = df.loc[:, "KO Gene"]
    for j, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        ax[2].scatter(df.iloc[:, 0][indx], df.iloc[:, 2][indx])
    for j, txt in enumerate(KO_genes):
        ax[2].annotate(txt, (df.iloc[j, 0], df.iloc[j, 2]))
    ax[2].set_xlabel("PC1 (" + str(round(pca_object.explained_variance_ratio_[0] * 100, 2)) + "%)")
    ax[2].set_ylabel("PC3 (" + str(round(pca_object.explained_variance_ratio_[2] * 100, 2)) + "%)")
    ax[2].set_title("PC3 vs PC1")
