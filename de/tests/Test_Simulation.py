#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''Add a test to the simulation of ODE. 
   The dimension of parameters(eps,w,alpha,beta) should be n_x, n_x*n_x, n_x, n_x.
   Expected results would be an array with N_t columns and nx rows.
   In addition, the relative concentration of knockout component should approach zero.
'''

import unittest
import numpy as np
from model import Model

class Test_ODE(unittest.TestCase):
    '''Test class for basic simulation of ODE systems'''
    def test_params(self):
        test1 = Model()
        test1.random_params()
        self.assertEqual(np.shape(test1.eps), (test1.n_x,))
        self.assertEqual(np.shape(test1.w), (test1.n_x,test1.n_x))
        self.assertEqual(np.shape(test1.alpha), (test1.n_x,))
        self.assertEqual(np.shape(test1.beta), (test1.n_x,))
        
    def test_simulation(self):
        test1 = Model()
        test1.random_params()
        test1.sim()
        self.assertTrue(isinstance(test1.sol, np.ndarray))
        self.assertEqual(len(test1.sol[:,0]), test1.N_t)
        self.assertEqual(len(test1.sol[0,:]), test1.n_x)
        self.assertEqual(int(test1.sol[50,0]), 0)


# In[4]:





# In[ ]:




