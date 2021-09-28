""" Methods to implement the factorization/fitting process. """

from typing import Union
import numpy as np
from tqdm import tqdm
from scipy.special import expit, logit
from .importData import importLINCS
from sklearn.linear_model import orthogonal_mp


alpha = 0.1


def calcW(data, eta, alphaIn):
    """
    Directly calculate w.

    Calculate an estimate for w based on data and current iteration of eta

    :param data: matrix or list of matrices representing a cell line's gene expression interactions with knockdowns
    :type data: Array or Array List
    param eta: vector representing overall pertubation effects of each gene of a cell line
    :type eta: Array
    param alphaIn: model parameter held constant due to steady-state approximation
    :type alphaIn: float
    output w: matrix representing gene-to-gene pertubation effects for either a singular cell line or multiple cell lines
    :type w: Array
    
    """
    if isinstance(data, np.ndarray):
        data = [data]

    return cost


def calcW(data: list, eta: list, alphaIn: float) -> np.ndarray:
    """
    Directly calculate w.

    Calculate an estimate for w based on data and current iteration of eta
    """
    for i, x in enumerate(data):
        U1 = np.copy(x)
        np.fill_diagonal(U1, 0.0)
        B1 = (x * alphaIn) / eta[i][:, np.newaxis]
        B1 = logit(np.clip(B1, 0.0001, 0.9999))

        if i == 0:
            U = U1
            B = B1
        else:
            U = np.concatenate((U, U1), axis=1)
            B = np.concatenate((B, B1), axis=1)

    w = np.linalg.lstsq(U.T, B.T, rcond=None)[0].T
    return w


def calcEta(data, w, alphaIn):
    """
    Calculate an estimate for eta based on data and current iteration of w

    :param data: matrix representing gene expression interactions with knockdowns
    :type data: Array or Array List
    param w: matrix representing gene-to-gene pertubation effects for either a singular cell line or multiple cell lines
    :type w: Array
    param alphaIn: model parameter held constant due to steady-state approximation
    :type alphaIn: float
    :output eta: vector representing overall perturbation effect of genes in each cell line
    :type eta: Array
    """

    eta = (alphaIn * data) / expit(w @ data)
    eta = np.nan_to_num(eta, nan=1.0, posinf=1e9)
    eta = np.clip(eta, 1e-9, 1e9)
    eta = gmean(eta, axis=1)
    return eta

def factorizeEstimate(data, tol=1e-9, maxiter=20):
    """ 
    Iteravely solve for w and eta list based on the data.

    :param data: matrix or list of matrices representing a cell line's gene expression interactions with knockdowns
    :type data: Array or Array List
    param tol: the minimum difference between two cost iteration values necessary to stop factorization process
    :type tol: float
    param maxiter: maximum amount of iterations for factorization process, given no tolerance break
    :type maxiter: int
    output w: finalized matrix representing gene-to-gene pertubation effects for either a singular cell line or multiple cell lines
    :type w: Array
    :output etas: finalized list of vectors representing overall perturbation effect of genes in each cell line
    :type etas: Array List
    """
    assert maxiter > 0
    if isinstance(data, np.ndarray):
        data = [data]

    w = np.zeros((data[0].shape[0], data[0].shape[0]))

    cost = np.inf

    # Use the data to try and initialize the parameters
    tq = tqdm(range(maxiter), delay=0.5)
    for _ in tq:
        etas = [calcEta(x, w, alpha) for x in data]
        w = calcW(data, etas, alpha)
        assert np.all(np.isfinite(w))
        assert w.shape == (data[0].shape[0], data[0].shape[0])

        cost = 0
        for jj in range(len(data)):
            cost += np.linalg.norm(etas[jj][:, np.newaxis]
                                   * expit(w @ U[jj]) - alpha * data[jj])

        if (costLast - cost) < tol:
            break

    if returnCost:
        return w, etas, cost

    return w, etas

def cellLineFactorization(cellLine):
    """ 
    Import a cell line, fit the model, and return the result.

    :param cellLine: name of the cell line for which we need to produce w and eta
    :type cellLine: string
    :output w: finalized matrix representing gene-to-gene pertubation effects for the cell line
    :type w: Array
    :output eta: finalized list of vectors representing overall perturbation effect of genes in each cell line
    :type eta: Array list
    :output annotation[0].tolist():
    :type annotation[0].tolist(): list
    """
    data, annotation = importLINCS(cellLine)
    w, eta = factorizeEstimate(data)
    return w, eta, annotation[0].tolist()

def commonGenes(annotation1, annotation2):
    """
    Uses annotation list to generate an array of common genes between two cell lines
    output

    :param annotation1: list of gene names from first cell line
    :type annotation1: list
    :param annotation2: list of gene names from second cell line
    :type annotation2: list
    :output indexlist_1: list of indices of the common genes for the first cell line
    :type indexlist_1: list
    :output indexlist_2: list of indices of the common genes for the first cell line
    :type indexlist_2: list

    """

    annotation1 = annotation1[0].tolist()
    annotation2 = annotation2[0].tolist()

    intersection = set(annotation1).intersection(annotation2)
    intersection_annotation = list(intersection)

    idx1 = np.array([ann1.index(x) for x in intersection], dtype=int)
    idx2 = np.array([ann2.index(x) for x in intersection], dtype=int)
    return idx1, idx2

def MatrixSubtraction(cellLine1, cellLine2):
    """Finds the w-matrices of two different cell lines and subtracts them.
    Then, calculates the norm of the original matrices as well as difference matrix
    
    :param cellLine1: name of the cell line for which we need to produce w1
    :type cellLine1: string
    :param cellLine2: name of the cell line for which we need to produce w2
    :type cellLine2: string
    :output norm1: norm of the w1-matrix produced using only common genes
    :type norm1: float
    :output norm2: norm of the w2-matrix produced using only common genes
    :type norm2: float
    :output diff_norm: norm of the matrix produced by subtracting w2_final and w1_final
    :type diff_norm: float
    :output w1_final: w1 matrix only accounting for common genes
    :type w1_final: Array
    :output w2_final: w2 matrix only accounting for common genes
    :type w2_final: Array

    """
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)
    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)
    w1, _ = factorizeEstimate(data1)
    w2, _ = factorizeEstimate(data2)

    w1 = w1[index_list1, :]
    w2 = w2[index_list2, :]
    w1 = w1[:, index_list1]
    w2 = w2[:, index_list2]
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

def mergedFitting(cellLine1, cellLine2):
    """
    Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix.
    
    :param cellLine1: name of the first cell line for which we need to produce w_shared
    :type cellLine1: string
    :param cellLine2: name of the second cell line for which we need to produce w_shared
    :type cellLine2: string
    :output w_shared: a matrix representing the gene-to-gene perturbation effects of both cell lines
    :type w_shared: array
    :output eta_list: list of vectors representing overall perturbation of genes in each cell line
    :type eta_list: array list
    """
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)
    index_list1, index_list2 = commonGenes(annotation1, annotation2)

    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)
    print(data1.shape)

    # Make shared
    data1 = data1[index_list1, :]
    data2 = data2[index_list2, :]
    data1 = data1[:, index_list1]
    data2 = data2[:, index_list2]
    shared_data = [data1, data2]

    return factorizeEstimate(shared_data)
