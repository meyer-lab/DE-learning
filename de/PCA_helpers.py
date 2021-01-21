"""Contains functions to perform PCA on knockout RNAseq data"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# ------------------------- Perform PCA using sklearn


def performPCA(data_in, num_components):
    """Function takes in parameter for number of components. Returns PCA object, fitted model"""
    pca_object = PCA(n_components=num_components)
    X_r = pca_object.fit_transform(normalize(data_in))
    return pca_object, X_r


# -------------------------- Create dataframe of PC scores returned from pca() and associated KO gene
def KOdataframe(data, X_r):
    """Function takes in parameters of original dataset to match names of knockouts and PC scores from pca().
    Returns dataframe associating name of knockout with each row of PC score and set of unique KO gene names."""
    KO_genes_unique = list(set(data.columns))
    df = pd.DataFrame(X_r)
    df["KO Gene"] = data.columns
    return df, KO_genes_unique
