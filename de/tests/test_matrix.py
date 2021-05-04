'''
Test the matrix for finding overlapping genes.
'''
import pytest
import numpy as np
from ..matrix import cellLineComparision
from ..factorization import cellLineFactorization

def test_cellLines():
    """ To test and confirm most genes are overlapping between cell lines. """
    cellLine1 = 'A375'
    cellLine2 = 'HT29'
    _, _, annotation1 = cellLineFactorization(cellLine1)
    _, _, annotation2 = cellLineFactorization(cellLine2)

    # assuming the function returns the list of shared genes between the two cell lines
    shared_annotation, _ = cellLineComparision(cellLine1, cellLine2)
    # make sure at least 50% of the genes in smaller cell line is shared between the two cell lines
    assert np.abs(len(shared_annotation)) >= 0.5 * len(np.min(len(annotation1), len(annotation2)))
