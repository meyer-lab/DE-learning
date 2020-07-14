'''
Add a test to the simulation of ODE.
The dimension of parameters(eps,w,alpha,beta) should be n_x, n_x*n_x, n_x, n_x.
Expected results would be an array with N_t columns and nx rows.
In addition, the relative concentration of knockout component should approach zero.
'''
import unittest
import numpy as np
from ..py_model import Model

class Test_ODE(unittest.TestCase):
    '''Test class for basic simulation of ODE systems'''
    def test_params(self):
        '''
        Test the dimension of parameters
        '''
        test1 = Model()
        p = test1.random_params()
        self.assertEqual(np.shape(p), (test1.n_x ** 2 + 2 * test1.n_x, ))
    def test_simulation(self):
        '''
        Test the simulation of ODE model
        '''
        test1 = Model()
        p = test1.random_params()
        t, sol = test1.py_sim(p)
        self.assertTrue(isinstance(sol, np.ndarray))
        self.assertEqual(len(test1.sol[:, 0]), test1.N_t)
        self.assertEqual(len(test1.sol[0, :]), test1.n_x)
