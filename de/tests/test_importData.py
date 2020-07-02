"""
Tests import of RNAseq knockout data results in dataframe with knockouts as columns, genes measured (rpm) as rows.
importRNAseqKO should return dataframe with individual replicate columns and rows for the entire genome.
formMatrix should return dataframe with all replicate columns averaged together, WT removed, and only rows for genes associated with a knockout.
"""
import unittest
import pandas as pd
from ..importData import importRNAseqKO, formMatrix

class TestModel(unittest.TestCase):
    """Test class for importing RNAseq knockout data file and forming matrix of knockouts"""

    def test_import(self):
        """Tests that file is successfully loaded into a DataFrame"""
        data = importRNAseqKO()
        self.assertTrue(isinstance(data, pd.DataFrame))

    def test_matrix(self):
        """Tests that matrix formed is an 83x84 DataFrame with matching knockouts/gene measurements on the diagonal"""
        matrix = formMatrix(importRNAseqKO())
        self.assertTrue(isinstance(matrix, pd.DataFrame))
        self.assertEqual(matrix.shape[0], 83) # there were 83 knockout models and thus associated genes
        self.assertEqual(matrix.shape[1], 84) # there is an extra column due to the negative control
        self.assertEqual(matrix.loc["NFAT5", "NFAT5"], matrix.iloc[42, 42]) # check diagonal is matched pairs