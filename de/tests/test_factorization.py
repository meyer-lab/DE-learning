'''
Test the factorization model.
'''
import pytest
import numpy as np
from scipy.special import expit
from ..factorization import MatrixSubtraction, cellLineComparision, factorizeEstimate, alpha, cellLineFactorization
from ..fitting import runOptim
from ..importData import formMatrix


def test_factorizeEstimate():
    """ Test that this runs successfully with reasonable input. """
    data = formMatrix()
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)

    w, eta = factorizeEstimate(data)
    assert w.shape == (data.shape[0], data.shape[0])
    assert eta.shape == (data.shape[0], )

    wLess, etaLess = factorizeEstimate(data, maxiter=1)
    costOne = np.linalg.norm(eta[:, np.newaxis] * expit(w @ U) - alpha * data)
    costTwo = np.linalg.norm(etaLess[:, np.newaxis] * expit(wLess @ U) - alpha * data)
    assert costOne < costTwo


@pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
def test_factorizeBlank(level):
    """ Test that if gene expression is flat we get a blank w. """
    data = np.ones((12, 12)) * level
    w, eta = factorizeEstimate(data, maxiter=2)

    np.testing.assert_allclose(w, 0.0, atol=1e-9)
    np.testing.assert_allclose(eta, 2 * level * alpha)


@pytest.mark.parametrize("sizze", [(8, 8), (12, 13), (15, 14)])
def test_fit(sizze):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.lognormal(size=sizze)
    outt = runOptim(data, niter=20, disp=False)
    assert np.all(np.isfinite(outt))

def test_cellLines():
    """ To test and confirm most genes are overlapping between cell lines. """
    cellLine1 = 'A375'
    cellLine2 = 'HT29'
    _, _, annotation1 = cellLineFactorization(cellLine1)
    _, _, annotation2 = cellLineFactorization(cellLine2)

    # assuming the function returns the list of shared genes between the two cell lines
    shared_annotation, _ = cellLineComparision(cellLine1, cellLine2)
    # make sure at least 50% of the genes in smaller cell line is shared between the two cell lines
    assert np.abs(len(shared_annotation)) >= 0.5 * np.min([len(annotation1), len(annotation2)])

def test_matrixSub():
    """To test if the matrices subtract properly and if the norm has a reasonble value"""
    cellLine1 = 'A375'
    cellLine2 = 'HT29'
    
    [matrix_1, matrix_2] = MatrixSubtraction(cellLine1, cellLine2)
    assert matrix_1.shape() == matrix_2.shape()

    w1, _, _ = cellLineFactorization(cellLine1)
    w2, _, _ = cellLineFactorization(cellLine2)
    test_norm1 = np.linalg.norm(w1)
    test_norm2 = np.linalg.norm(w2)


    