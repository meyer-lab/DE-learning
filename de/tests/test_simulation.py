'''
Add a test to the simulation of ODE.
The dimension of parameters(eps,w,alpha,beta) should be n_x, n_x*n_x, n_x, n_x.
Expected results would be an array with N_t columns and nx rows.
In addition, the relative concentration of knockout component should approach zero.
'''
import unittest
import numpy as np
from ..model import Model

class Test_ODE(unittest.TestCase):
    '''Test class for basic simulation of ODE systems'''
    def test_params(self):
        '''
        Test the dimension of parameters
        '''
        test1 = Model()
        p, beta = test1.random_params(1000)
        self.assertEqual(np.shape(p), (test1.n_x ** 2 + 2 * test1.n_x, ))
        self.assertEqual(np.shape(beta), (test1.n_x, test1.N))

    def test_simulation(self):
        '''
        Test the simulation of ODE model
        '''
        test1 = Model()
        p, beta = test1.random_params(1000)
        sol = test1.sim(p, beta, 48)
        self.assertTrue(isinstance(sol, np.ndarray))
        self.assertEqual(sol.shape, (test1.n_x, test1.N))
