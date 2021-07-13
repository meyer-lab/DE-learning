import unittest
import numpy as np
import pandas as pd
from ..importData import importmelanoma

class TestModel(unittest.TestCase):
    def test_merge(self):
        """Tests that merged DataFrame has three columns: target, meanlFC, and Rcolonies_lFC"""
        matrix = importmelanoma()
        self.assertTrue(isinstance(matrix, pd.DataFrame))
        self.assertEqual(matrix.shape[1], 3)
        assert len(matrix) <= 88 #merged matrix has fewer genes than input files

    def test_overlap(self):
        """To test and confirm all targets/genes are found in both x_data and y_data"""
        x_data = importmelanoma()
        y_data = importmelanoma()
        _, _, annotation1 = x_data
        _, _, annotation2 = y_data
        data = importmelanoma()
        assert np.abs(len(data)) == np.min([len(annotation1), len(annotation2)])
