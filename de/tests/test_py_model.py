'''
Add a test to the simulation of ODE.
The dimension of parameters(eps,w,alpha,beta) should be n_x, n_x*n_x, n_x, n_x.
Expected results would be an array with N_t columns and nx rows.
In addition, the relative concentration of knockout component should approach zero.
'''
import unittest
import numpy as np
from ..py_model import graph, random_params, scatterplot

class TestPyModel(unittest.TestCase):
    '''Test class for basic simulation of ODE systems'''
    def test_params(self):
        '''
        Test the dimension of parameters
        '''
        p = random_params()
        self.assertEqual(np.shape(p), (83 ** 2 + 2 * 83, ))

    def test_graph(self):
        '''
        Test the graph function
        '''
        t_test = np.arange(100)
        sol_test = np.random.normal(0.01, 1.0, size=(100, 83))
        graph("random test", t_test, sol_test)

    def test_scatterplot(self):
        '''
        Test the scatterplot function
        '''
        exp_rand = np.random.normal(0.01, 1.0, size=(83, 84))
        model_rand = np.random.normal(0.01, 1.0, size=(83, 84))
        scatterplot(exp_rand, model_rand)
