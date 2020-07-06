'''
Basic Model simulation
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from ODE import solver

class Model(object):
    def __init__(self, n_x=83, pert=50, index=np.loadtxt('./data/node_Index.csv', delimiter=',', dtype=str), dt=0.1, N_t=100, filename='./figures/simulation_1.jpg'):
        self.n_x = n_x #The number of components involved
        self.x0 = np.ones(n_x) #Initial values
        self.pert = pert # The strength of pertubation
        self.index = index # The gene name for each component
        self.dt, self.N_t = dt, N_t # Time series
        self.filename = filename #Path for saving simulation result

    def graph(self):
        '''Function receives simulation of ODE and the number of components involved, then generate the graph'''
        fig = plt.figure(figsize=(100,300.0))
        for i in range(self.n_x):
            axes = fig.add_subplot(math.ceil(self.n_x/5),5,i+1)
            axes.plot(self.t,self.sol[:,i],'r')
            axes.set_title(self.index[i])
        plt.xlabel('t')
        plt.ylabel('relative concentration')   # x(t)/x(0)
        fig.tight_layout()
        plt.savefig(self.filename)
        plt.show()
        
    def sim(self):
        '''Run the ODE model'''
        self.t, self.sol = solver(self.n_x,self.x0,self.eps,self.w,self.alpha,self.beta,self.dt,self.N_t)
        
    def random_params(self):
        '''Randomly initialize the parameters based on the number of components and pertubation strength.
           only consider one knock-out condition here.
        '''
        self.eps = np.random.normal(1,1.0,size=(self.n_x))
        W = np.random.normal(0.01,1.0,size=(self.n_x,self.n_x))
        W_mask = (1.0 - np.diag(np.ones([self.n_x]))) #remove self-interaction
        self.w = W_mask*W
        self.alpha = np.random.normal(2,1.0,size=(self.n_x))
        self.beta = np.ones(self.n_x)
        self.beta[0] = self.pert
