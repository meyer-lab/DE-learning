import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.decomposition import PCA
from importData import importRNAseqKO

data = importRNAseqKO()

#------------------------- Perform PCA using sklearn
def pca(data, num_components):
    "Function takes in parameter for number of components. Returns list containing: [PCA object, fitted model]"
    pca = PCA(n_components=num_components)
    X_r = pca.fit(data).transform(data)
    return [pca, X_r]

#------------------------- Calculate cumulative variance based on number of PCs included and create r2x plot
def r2x(num_components, pca):
    total_variance = np.array([])
    tot = 0.0
    for i in range(0,num_components):
        tot += pca.explained_variance_ratio_[i]
        total_variance = np.append(total_variance, tot)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Number of PCs', fontsize = 15)
    ax.set_ylabel('% Variance', fontsize = 15)
    plt.xticks(np.arange(num_components+1))
    plt.plot(list(range(1,num_components+1)),total_variance)
    ax.grid()
    plt.show()
    
#------------------------- Create PC plots with PCs 1-4 (no color or labelling)
def pc2Dplotting(pca_list):
    X_r = pca_list[1]
    fig = plt.figure(figsize=(25,20))
    # PC1 vs PC2
    ax = plt.subplot(231)
    ax.set_xlabel("PC1", fontsize = 15)
    ax.set_ylabel("PC2", fontsize = 15)
    plt.scatter(X_r[:,0], X_r[:,1])
    ax.grid()

    # PC1 vs PC3
    ax = plt.subplot(232)
    ax.set_xlabel("PC1", fontsize = 15)
    ax.set_ylabel("PC3", fontsize = 15)
    plt.scatter(X_r[:,0], X_r[:,2])
    ax.grid()

    # PC1 vs PC4
    ax = plt.subplot(233)
    ax.set_xlabel("PC1", fontsize = 15)
    ax.set_ylabel("PC4", fontsize = 15)
    plt.scatter(X_r[:,0], X_r[:,3])
    ax.grid()

    # PC2 vs PC3
    ax = plt.subplot(234)
    ax.set_xlabel("PC2", fontsize = 15)
    ax.set_ylabel("PC3", fontsize = 15)
    plt.scatter(X_r[:,1], X_r[:,2])
    ax.grid()

    # PC2 vs PC4
    ax = plt.subplot(235)
    ax.set_xlabel("PC1", fontsize = 15)
    ax.set_ylabel("PC3", fontsize = 15)
    plt.scatter(X_r[:,1], X_r[:,3])
    ax.grid()

    # PC3 vs PC4
    ax = plt.subplot(236)
    ax.set_xlabel("PC3", fontsize = 15)
    ax.set_ylabel("PC4", fontsize = 15)
    plt.scatter(X_r[:,2], X_r[:,3])
    ax.grid()