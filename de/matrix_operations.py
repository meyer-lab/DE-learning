""" This file contains operations on data matrices. """
import numpy as np
from .importData import importLINCS
from .factorization import factorizeEstimate


def MatrixSubtraction(cellLine1, cellLine2):
    """Finds the w-matrices of two different cell lines.
    Calculates the norm of the original matrices and their difference.
    """
    data1, annotation1 = importLINCS(cellLine1)
    data2, annotation2 = importLINCS(cellLine2)
    indexes = commonGenes(annotation1, annotation2)
    index_list1, index_list2 = indexes[0], indexes[1]
    w1, _ = factorizeEstimate(data1)
    w2, _ = factorizeEstimate(data2)

    w1 = w1[index_list1, index_list1]
    w2 = w2[index_list2, index_list2]
    assert w1.shape == w2.shape
    assert w1.shape == (index_list1.size, index_list1.size)

    norm1 = np.linalg.norm(w1)
    print(f"Norm1: {norm1}")
    norm2 = np.linalg.norm(w2)
    print(f"Norm2: {norm2}")
    diff_norm = np.linalg.norm(w2 - w1)
    print(f"Difference norm: {diff_norm}")

    w1shuff = w1.copy()
    w2shuff = w2.copy()
    np.random.shuffle(w1shuff)
    np.random.shuffle(w2shuff)
    shufnorm = np.linalg.norm(w2shuff - w1shuff)
    print(f"Shuffled norm: {shufnorm}")
    return norm1, norm2, diff_norm, shufnorm, w1, w2
