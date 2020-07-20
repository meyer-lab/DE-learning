"""
This creates Figure 1: PCA plots
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..PCA_helpers import performPCA, KOdataframe
from ..importData import importRNAseqKO

def prepData():
    """ Load RNAseq data then average replicates and negative controls """
    d = importRNAseqKO()
    data_prepped = d.groupby(by=d.columns, axis=1).mean()
    data_prepped['neg'] = data_prepped[['neg01', 'neg10']].mean(axis=1)
    for i in range(1, 10):
        data_prepped = data_prepped.drop(["neg0"+str(i)], axis=1)
    data_prepped = data_prepped.drop(["neg10"], axis=1)
    return data_prepped

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 4), (1, 3))
    
    # Perform PCA
    data = prepData()
    pca_object, X_r = performPCA(data.T, 25)
    
    # Create dataframe for plotting
    df, KO_genes_unique = KOdataframe(data, X_r)
    
    # Make subplots
    figureMaker(ax, pca_object, df, KO_genes_unique)
    
    # Add subplot labels
    subplotLabel(ax)

    return f

def figureMaker(ax, pca_object, df, KO_genes_unique):
    """ Makes 3 panels: (A) R2X, (B) PC2 vs PC1, (C) ? """
    
    # Plot R2X
    i = 0
    total_variance = np.array([])
    tot = 0.0
    for j in range(0, 25):
        tot += pca_object.explained_variance_ratio_[j] * 100
        total_variance = np.append(total_variance, tot)
    ax[i].set_xlabel("Number of Components")
    ax[i].set_ylabel("% Variance")
    ax[i].set_xticks(np.arange(1, 26, 2))
    ax[i].plot(list(range(1, 26)), total_variance)
    ax[i].set_title("R2X")
    
    # Plot PC2 vs PC1
    i += 1
    KO_genes = df.loc[:, "KO Gene"]
    for j, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        ax[i].scatter(df.iloc[:, 0][indx], df.iloc[:, 1][indx], s=8)
    for j, txt in enumerate(KO_genes):
        ax[i].annotate(txt, (df.iloc[j, 0], df.iloc[j, 1]), fontsize=6)
    ax[i].set_xlabel("PC1 (" + str(round(pca_object.explained_variance_ratio_[0]*100, 2)) + "%)")
    ax[i].set_ylabel("PC2 (" + str(round(pca_object.explained_variance_ratio_[1]*100, 2)) + "%)")
    ax[i].set_title("PC2 vs PC1")
    
    # Plot PC3 vs PC1
    i += 1
    KO_genes = df.loc[:, "KO Gene"]
    for j, gene in enumerate(KO_genes_unique):
        indx = df["KO Gene"] == gene
        ax[i].scatter(df.iloc[:, 0][indx], df.iloc[:, 2][indx], s=8)
    for j, txt in enumerate(KO_genes):
        ax[i].annotate(txt, (df.iloc[j, 0], df.iloc[j, 2]), fontsize=6)
    ax[i].set_xlabel("PC1 (" + str(round(pca_object.explained_variance_ratio_[0]*100, 2)) + "%)")
    ax[i].set_ylabel("PC3 (" + str(round(pca_object.explained_variance_ratio_[2]*100, 2)) + "%)")
    ax[i].set_title("PC3 vs PC1")
