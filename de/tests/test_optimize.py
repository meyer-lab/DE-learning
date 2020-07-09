"""
Tests residual function used to find difference between ODE model and RNAseq data. Then tests that least_squares is reducing SSE as it iterates.
The residual function should return a 1D ndarray of 84 * 83 elements.
least_squares should return a set of parameters that leads to SSE lower than the randomly intialized parameters.
"""
import unittest
import numpy as np
from scipy.optimize import least_squares
from ..model import Model
from ..importData import importRNAseqKO, formMatrix
from ..optimize import residual_fun

class TestModel(unittest.TestCase):
    """Test class for residual_fun and use of scipy.optimize.least_squares"""

    def test_residual_fun(self):
        data = importRNAseqKO()
        exp = formMatrix(data)
        m = Model()
        params_i = m.random_params()
        residuals = residual_fun(params_i, m, exp)
        self.assertEqual((84 * 83, ), residuals.shape)

    def test_least_squares(self):
        data = importRNAseqKO()
        exp = formMatrix(data)
        m = Model()
        params_i = m.random_params()
        residuals = residual_fun(params_i, m, exp)
        cost_i = 0.5 * sum(residuals ** 2)
        opt_params, cost = least_squares(residual_fun, params_i, args=(m, exp), max_nfev=2)
        self.assertTrue(cost < cost_i)
