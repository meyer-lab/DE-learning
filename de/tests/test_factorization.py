'''
Test the factorization model.
'''
import pytest
import numpy as np
import numpy.ma as ma
from scipy.special import expit
from ..factorization import factorizeEstimate, alpha, cellLineComparison, MatrixSubtraction, cellLineFactorization, cross_val
from ..fitting import runOptim, impute
from ..importData import ImportMelanoma, importLINCS


def test_factorizeEstimate():
    """ Test that this runs successfully with reasonable input. """
    data = ImportMelanoma()
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
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)

    # assuming the function returns the list of shared genes between the two cell lines
    shared_annotation, _ = cellLineComparison(cellLine1, cellLine2)
    # make sure at least 50% of the genes in smaller cell line is shared between the two cell lines
    assert np.abs(len(shared_annotation)) >= 0.5 * np.min([len(annotation1), len(annotation2)])

def test_matrixSub():
    """To test if the matrices subtract properly and if the norm has a reasonable value"""
    cellLine1 = 'A375'
    cellLine2 = 'HT29'
    
    norm1, norm2, diff_norm, _, _ = MatrixSubtraction(cellLine1, cellLine2)

    assert diff_norm != norm1
    assert diff_norm != norm2

def test_crossval_Melanoma():
    """ Tests the cross val function that creates the train and test data. """
    data = ImportMelanoma()
    train_X, test_X = cross_val(data)
    full_X = impute(train_X)

    print(ma.corrcoef(ma.masked_invalid(full_X.flatten()), ma.masked_invalid(test_X.flatten())))


def test_crossval_LINCS(cellLine):
    """ Tests the cross val function that creates the train and test data. """

    data, _ = importLINCS(cellLine)
    U = np.copy(data)
    np.fill_diagonal(U, 0.0)
    
    data_corr = []
    U_corr = []
    print('check1')
    x = range(0,10)

    for j in x:
        train_X, test_X = cross_val(data)
        full_X = impute(train_X)
        data_corr.append(ma.corrcoef(ma.masked_invalid(full_X.flatten()), ma.masked_invalid(test_X.flatten()))[1,0])
        
        train_Y, test_Y = cross_val(U)
        full_Y = impute(train_Y)
        U_corr.append(ma.corrcoef(ma.masked_invalid(full_Y.flatten()), ma.masked_invalid(test_Y.flatten()))[1,0])

    data_corr_avg = np.average(data_corr)
    data_corr_std = np.std(data_corr)
    U_corr_avg = np.average(U_corr)
    U_corr_std = np.std(U_corr)
        
    return data_corr_avg, U_corr_avg, data_corr_std, U_corr_std
