"""
Tests import of RNAseq knockout data results in dataframe with knockouts as columns, genes measured (rpm) as rows.
importRNAseqKO should return dataframe with individual replicate columns and rows for the entire genome.
ImportMelanoma should return dataframe with all replicate columns averaged together, WT removed, and only rows for genes associated with a knockout.
"""
import unittest
import pytest
import pandas as pd
import numpy as np
from ..importData import importRNAseqKO, ImportMelanoma, prepData, importLINCS, importgenes


class TestModel(unittest.TestCase):
    """Test class for importing RNAseq knockout data file and forming matrix of knockouts"""

    def test_import(self):
        """Tests that file is successfully loaded into a DataFrame"""
        data = importRNAseqKO()
        self.assertTrue(isinstance(data, pd.DataFrame))

    def test_matrix(self):
        """Tests that matrix formed is an 83x84 DataFrame with matching knockouts/gene measurements on the diagonal"""
        matrix, _ = ImportMelanoma()
        self.assertTrue(isinstance(matrix, np.ndarray))
        self.assertEqual(matrix.shape[0], 83)  # there were 83 knockout models and thus associated genes
        self.assertEqual(matrix.shape[1], 84)  # there is an extra column due to the negative control

    def test_prep(self):
        """Tests that a DataFrame is formed with a row for every gene and 85 columns to represent models"""
        data = prepData()
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertEqual(data.shape, (63677, 85))

    def test_merge(self):
        """Tests that merged DataFrame has three columns: target, meanlFC, and Rcolonies_lFC"""
        matrix = importgenes()
        self.assertTrue(isinstance(matrix, pd.DataFrame))
        self.assertEqual(matrix.shape[1], 3)


@pytest.mark.parametrize("cellLine", ["A375", "A549", "HA1E", "MCF7"])
def test_load(cellLine):
    """Tests that a DataFrame is formed with a row for every gene and 85 columns to represent models"""
    data, annotation = importLINCS(cellLine)
    assert data.ndim == 2
    assert data.shape[0] == len(annotation)
