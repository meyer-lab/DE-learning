"""Contains functions to perform PCA on knockout RNAseq data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

#------------------------- Perform PCA using sklearn
def performPCA(data_in, num_components):
    "Function takes in parameter for number of components. Returns list containing: [PCA object, fitted model]"
    pca_object = PCA(n_components=num_components)
    X_r = pca_object.fit_transform(normalize(data_in))
    return [pca_object, X_r]

#------------------------- Calculate cumulative variance based on number of PCs included and create r2x plot
def r2x(num_components, pca_object, fname):
    "Function takes in parameters for number of components, PCA object returned from pca(), and filename for plot image and creates r2x plot"
    total_variance = np.array([])
    tot = 0.0
    for i in range(0, num_components):
        tot += pca_object.explained_variance_ratio_[i]
        total_variance = np.append(total_variance, tot)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of PCs", fontsize=15)
    ax.set_ylabel("% Variance", fontsize=15)
    plt.xticks(np.arange(num_components+1))
    plt.plot(list(range(1, num_components+1)), total_variance)
    ax.grid()
    plt.savefig("PCA Images/"+fname)

#-------------------------- Create dataframe of PC scores returned from pca() and associated KO gene
def KOdataframe(data, X_r):
    """Function takes in parameters of original dataset to match names of knockouts and PC scores from pca().
    Returns list containing dataframe associating name of knockout with each row of PC score and set of unique KO gene names."""
    KO_genes_unique = list(set(data.columns))
    df = pd.DataFrame(X_r)
    df["KO Gene"] = data.columns
    return [df, KO_genes_unique]

#-------------------------- Create PC plots
def plottingPCs(KO_genes_list, fname):
    """Function takes in list returned from KOdataframe() containing PC scores and set of unique KO genes, filename to save plots as.
    Creates 2x3 figure displaying 2d plot comparisons of first 4 PCs. Points are colored by knockout gene and annotated."""
    df = KO_genes_list[0]
    KO_genes_unique = KO_genes_list[1]
    KO_genes = df.loc[:, "KO Gene"]

    # Set the color map to match the number of species
    z = range(1, len(KO_genes_unique))
    rainbow = plt.get_cmap("rainbow")
    cNorm = colors.Normalize(vmin=0, vmax=len(KO_genes_unique))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)

    fig = plt.figure(figsize=(50, 40))
    # PC1 vs PC2
    ax = plt.subplot(231)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 0][indx], df.iloc[:, 1][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 0], df.iloc[i, 1]), fontsize=6)

    # PC1 vs PC3
    ax = plt.subplot(232)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC3", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 0][indx], df.iloc[:, 2][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 0], df.iloc[i, 2]), fontsize=6)

    # PC1 vs PC4
    ax = plt.subplot(233)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC4", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 0][indx], df.iloc[:, 3][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 0], df.iloc[i, 3]), fontsize=6)

    # PC2 vs PC3
    ax = plt.subplot(234)
    ax.set_xlabel("PC2", fontsize=15)
    ax.set_ylabel("PC3", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 1][indx], df.iloc[:, 2][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 1], df.iloc[i, 2]), fontsize=6)

    # PC2 vs PC4
    ax = plt.subplot(235)
    ax.set_xlabel("PC2", fontsize=15)
    ax.set_ylabel("PC4", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 1][indx], df.iloc[:, 3][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 1], df.iloc[i, 3]), fontsize=6)

    # PC3 vs PC4
    ax = plt.subplot(236)
    ax.set_xlabel("PC3", fontsize=15)
    ax.set_ylabel("PC4", fontsize=15)
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        plt.scatter(df.iloc[:, 2][indx], df.iloc[:, 3][indx], s=10, color=scalarMap.to_rgba(i), label=gene)
    ax.grid()
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, 2], df.iloc[i, 3]), fontsize=6)
    plt.savefig("PCA Images/"+fname)
#-------------------------- Create individual plots for each knockout gene's replicates on PCy v PCx
def plottingPCreplicates(KO_list, PCx, PCy):
    """Function takes in 3 parameters: list returned from KOdataframe(), and integers for the 2 PCs to be plotted in order (x,y).
    The function creates a 2x47 plot, each subplot representing 1 knockout gene on the same scale."""
    df = KO_list[0]
    KO_genes_unique = KO_list[1]
    fig, axes = plt.subplots(nrows=47, ncols=2, figsize=(30, 180))
    fig.suptitle("KO Genes PC"+str(PCy)+" vs PC"+str(PCx), fontsize=18)
    for ax, name in zip(axes.flatten(), KO_genes_unique):
        indx = df["KO Gene"] == name
        ax.scatter(df.iloc[:, (PCx-1)][indx], df.iloc[:, (PCy-1)][indx], s=30)
        ax.grid()
        ax.set_title(name)
        ax.set_xlim((-20000, 7000))
        ax.set_ylim((-25000, 18000))
        fig = ax.get_figure()
    fig.text(0.5, 0.04, "PC"+str(PCx), ha="center", va="center")
    fig.text(0.06, 0.5, "PC"+str(PCy), ha="center", va="center", rotation="vertical")
    fig.tight_layout()
    fig.subplots_adjust(top=0.975)
    plt.savefig("PCA Images/"+"PC"+str(PCy)+"vPC"+str(PCx)+"_all.png")
