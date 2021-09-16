'''
Test the factorization model.
'''
import pytest
import numpy as np
from numpy import ma
from ..factorization import factorizeEstimate, alpha, commonGenes, mergedFitting
from ..impute import impute, split_data, repeatImputation
from ..importData import ImportMelanoma, importLINCS


def test_factorizeEstimate():
    """ Test that this runs successfully with reasonable input. """
    data = ImportMelanoma()

    w, eta, costOne = factorizeEstimate(data, returnCost=True)
    assert w.shape == (data.shape[0], data.shape[0])
    assert eta[0].shape == (data.shape[0], )

    _, _, costTwo = factorizeEstimate(data, maxiter=1, returnCost=True)
    assert costOne < costTwo


@pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
def test_factorizeBlank(level):
    """ Test that if gene expression is flat we get a blank w. """
    data = np.ones((12, 12)) * level
    w, eta = factorizeEstimate(data, maxiter=2)

    np.testing.assert_allclose(w, 0.0, atol=1e-9)
    np.testing.assert_allclose(eta, 2 * level * alpha)


def test_cellLines():
    """ To test and confirm most genes are overlapping between cell lines. """
    cellLine1 = 'A375'
    cellLine2 = 'HT29'
    _, annotation1 = importLINCS(cellLine1)
    _, annotation2 = importLINCS(cellLine2)

    mergedFitting(cellLine1, cellLine2)

    # assuming the function returns the list of shared genes between the two cell lines
    shared_annotation, _ = commonGenes(annotation1, annotation2)
    # make sure at least 50% of the genes in smaller cell line is shared between the two cell lines
    assert np.abs(len(shared_annotation)) >= 0.5 * np.min([len(annotation1), len(annotation2)])


def test_mergedFitting():
    """ To test if the fitting works on multiple cell lines and the shared cost has a reasonable value. """
    data = ImportMelanoma()
    w1, eta_list1 = factorizeEstimate(data)
    eta1 = eta_list1[0]

    w2, eta_list2 = factorizeEstimate([data, data])
    eta2 = eta_list2[0]

    # Both etas should be the same
    np.testing.assert_allclose(eta_list2[0], eta_list2[1], rtol=0.1)
    np.testing.assert_allclose(eta1, eta2, rtol=0.1)

    # w should be identical
    np.testing.assert_allclose(w1, w2, atol=1.5)


def test_crossval_Melanoma():
    """ Tests the cross val function that creates the train and test data. """
    data = ImportMelanoma()
    train_X, test_X = split_data(data)
    full_X = impute(train_X)

    print(ma.corrcoef(ma.masked_invalid(full_X.flatten()), ma.masked_invalid(test_X.flatten())))
