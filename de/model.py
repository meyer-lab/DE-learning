'''
Basic Model simulation
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from ODE import solver, jacobian_autograd

class Model():
    '''
    The class for basic model simulation
    '''
    def __init__(self, n_x=83, N=84, pert=50, index='./data/node_Index.csv', dt=0.1, N_t=100, save_path='./figures/simulation_1.jpg', expdata_path='./data/exp_data.csv'):
        self.n_x = n_x #The number of components involved
        self.N = N #The number of knockout conditions
        self.x0 = np.ones(n_x) #Initial values
        self.pert = pert # The strength of pertubation
        self.index = np.loadtxt(index, delimiter=',', dtype=str) # The gene name for each component
        self.dt, self.N_t = dt, N_t # Time step and total length of time for generating time series
        self.save_path = save_path #Path for saving simulation result
        self.expdata_path = expdata_path #Path for inputing experiment data
        self.t = None #Time series
        self.sol = None #Simulation of ODE
        self.eps = None #Value that bound the saturation effect
        self.w = None #Interaction strength
        self.alpha = None #Degradation rate
        self.beta = None #Knock-out effects
        self.sse = None # The sum of square error across all measurements
        self.jacb = None # The jacobian matrix of the system

    def graph(self, cond):
        '''Function receives simulation of ODE and the number of components involved, then generate the graph.
           Parameters: cond = assigns a specific knockout condition.
        '''
        fig = plt.figure(figsize=(100, 300.0))
        for i in range(self.n_x):
            axes = fig.add_subplot(math.ceil(self.n_x/5), 5, i+1)
            axes.plot(self.t, self.sol[cond, :, i], 'r')
            axes.set_title(self.index[i])
        plt.xlabel('t')
        plt.ylabel('relative concentration')   # x(t)/x(0)
        fig.tight_layout()
        plt.savefig(self.save_path)
        plt.show()
    def sim(self):
        '''Run the ODE model'''
        self.t, self.sol = solver(self.n_x, self.N, self.x0, self.eps, self.w, self.alpha, self.beta, self.dt, self.N_t)
    def jac(self):
        ''' Obtain the jacobian matrix of the system'''
        self.jacb = jacobian_autograd(self.sol, self.eps, self.w, self.alpha, self.beta, self.N, self.n_x)
    def random_params(self):
        '''
        Randomly initialize the parameters based on the number of components and pertubation strength.
        only consider one knock-out condition here.
        '''
        self.eps = np.abs(np.random.normal(1, 1.0, size=(self.n_x)))
        W = np.random.normal(0.01, 1.0, size=(self.n_x, self.n_x))
        W_mask = (1.0 - np.diag(np.ones([self.n_x]))) #remove self-interaction
        self.w = W_mask*W
        self.alpha = np.abs(np.random.normal(2, 1.0, size=(self.n_x)))
        beta = 1 + (np.diag(self.pert*np.ones(self.n_x)-1))
        neg = np.ones((83))
        self.beta = np.insert(beta, 83, values=neg, axis=1)
    def comparison(self):
        '''
        Compute the sum of square error across all the knock-out measurements
        '''
        x_exp = np.loadtxt(self.expdata_path, delimiter=',')
        x_sim = np.transpose(self.sol[:, -1, :])
        self.sse = np.sum(np.square(x_sim - x_exp))
    def scatterplot(self):
        '''
        Create scatterplot of model data vs experimental data
        '''
        rainbow = plt.get_cmap("rainbow")
        cNorm = colors.Normalize(vmin=0, vmax=self.n_x)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
        self.save_path = "./figures/modelvsdata_scatter.jpg"
        x_exp = np.loadtxt(self.expdata_path, delimiter=',')
        x_sim = np.transpose(self.sol[:, -1, :])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(self.n_x):
            plt.scatter(x_exp[i, :], x_sim[i, :], s=10, color=scalarMap.to_rgba(i), label=self.index[i])
        plt.xlabel('RNAseq Data')
        plt.ylabel('Model Solution at t = 48 hours')
        plt.title("Model vs Data at t=48 hours")
        fig.tight_layout()
        plt.savefig(self.save_path)
        plt.show()
