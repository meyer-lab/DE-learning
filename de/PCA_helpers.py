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
    """Function takes in parameter for number of components. Returns PCA object, fitted model"""
    pca_object = PCA(n_components=num_components)
    X_r = pca_object.fit_transform(normalize(data_in))
    return pca_object, X_r

#------------------------- Calculate cumulative variance based on number of PCs included and create r2x plot
def r2x(num_components, pca_object):
    """Function takes in parameters for number of components, PCA object returned from pca(), and filename for plot image and creates r2x plot"""
    total_variance = np.array([])
    tot = 0.0
    for i in range(0, num_components):
        tot += pca_object.explained_variance_ratio_[i]
        total_variance = np.append(total_variance, tot)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of Components", fontsize=15)
    ax.set_ylabel("% Variance", fontsize=15)
    plt.xticks(np.arange(num_components+1))
    plt.plot(list(range(1, num_components+1)), total_variance)
    ax.grid()

#-------------------------- Create dataframe of PC scores returned from pca() and associated KO gene
def KOdataframe(data, X_r):
    """Function takes in parameters of original dataset to match names of knockouts and PC scores from pca().
    Returns dataframe associating name of knockout with each row of PC score and set of unique KO gene names."""
    KO_genes_unique = list(set(data.columns))
    df = pd.DataFrame(X_r)
    df["KO Gene"] = data.columns
    return df, KO_genes_unique

#-------------------------- Create PC plots
def plottingPC(df, KO_genes_unique, PCnums, ax):
    '''Function takes in dataframe of PC scores, set of unique KO genes, array of list of PCs to be plotted (x, y), and axes.
    Creates figure displaying 2d plot comparison of PCs. Points are annotated by gene.'''
    KO_genes = df.loc[:, "KO Gene"]
    for i, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        ax.scatter(df.iloc[:, (PCnums[0] - 1)][indx], df.iloc[:, (PCnums[1] - 1)][indx], s=8)
    for i, txt in enumerate(KO_genes):
        ax.annotate(txt, (df.iloc[i, (PCnums[0] - 1)], df.iloc[i, (PCnums[1] - 1)]), fontsize=6)

#-------------------------- Create individual plots for each knockout gene's replicates on PCy v PCx
def plottingPCreplicates(KO_list, PCx, PCy):
    '''Function takes in 3 parameters: list returned from KOdataframe(), and integers for the 2 PCs to be plotted in order (x,y).
    The function creates a 2x47 plot, each subplot representing 1 knockout gene on the same scale.'''
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
