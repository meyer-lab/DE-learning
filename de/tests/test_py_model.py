'''
Add a test to the simulation of ODE.
The dimension of parameters(eps,w,alpha,beta) should be n_x, n_x*n_x, n_x, n_x.
Expected results would be an array with N_t columns and nx rows.
In addition, the relative concentration of knockout component should approach zero.
'''
import unittest
import numpy as np
from ..py_model import random_params

class TestPyModel(unittest.TestCase):
    '''Test class for basic simulation of ODE systems'''
    def test_params(self):
        '''
        Test the dimension of parameters
        '''
        p = random_params()
        self.assertEqual(np.shape(p), (83 ** 2 + 2 * 83, ))
