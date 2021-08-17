""" Contains functions to perform PCA on knockout RNAseq data. """

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def performPCA(data_in, num_components):
    """ Function takes in parameter for number of components. Returns PCA object, fitted model. """
    pca_object = PCA(n_components=num_components)
    X_r = pca_object.fit_transform(normalize(data_in))
    return pca_object, X_r
