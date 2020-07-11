'''
Basic Model simulation
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from .ODE import solver, jacobian_autograd

class Model():
    '''
    The class for basic model simulation
    '''
    def __init__(self, n_x=83, N=84, N_t = 48):
        self.n_x = n_x #The number of components/genes involved
        self.N = N #The number of knockout conditions
        self.x0 = np.ones(n_x) #Initial values
        self.N_t = N_t #Total length of simulation time

    def graph(self, cond, t, sol, save_path):
        '''Function receives simulation of ODE, the number of components involved, time series, model solution, and path at which to save image. 
        Generates graphs of the solution over time.
        '''
        # TODO: Will we be able to graph over time given we are only solving for 1 timepoint?
        fig = plt.figure(figsize=(100, 300.0))
        for i in range(self.n_x):
            axes = fig.add_subplot(math.ceil(self.n_x/5), 5, i+1)
            axes.plot(t, sol[cond, :, i], 'r')
            axes.set_title(cond)
        plt.xlabel('t')
        plt.ylabel('Expression Level')   # x(t)/x(0)
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def sim(self, params, beta_in):
        '''Function receives simulation of ODE, 1D array of optimizable parameters, array of beta parameter, and time series.
        Runs the ODE model and returns solution at last timepoint.'''
        sol = solver(self.n_x, self.N, self.x0, self.N_t, params, beta_in)
        return np.transpose(sol)
    
    def random_params(self, pert):
        '''
        Function receives simulation of ODE and perturbation strength and randomly initialize the parameters based on the number of components.
        Returns 1D array of parameters eps (n_x), w (n_x ** 2), and alpha (n_x) as well as beta
        '''
        eps = np.abs(np.random.normal(1, 1.0, size=(self.n_x)))
        W = np.random.normal(0.01, 1.0, size=(self.n_x, self.n_x))
        W_mask = (1.0 - np.diag(np.ones([self.n_x]))) #remove self-interaction
        w = W_mask*W
        alpha = np.abs(np.random.normal(2, 1.0, size=(self.n_x)))
        beta = 1 + (np.diag(pert*np.ones(self.n_x)-1))
        neg = np.ones((self.n_x))
        beta = np.insert(beta, self.n_x, values=neg, axis=1)
        # Combine all parameters into p
        p = np.concatenate([eps, w.flatten(), alpha])
        return p, beta

    def scatterplot(self, exp_data, model_data, save_path):
        '''
        Function receives simulation of ODE, matrix of RNAseq data, matrix of model solution, and save path for image.
        Creates scatterplot of model data vs experimental data
        '''
        rainbow = plt.get_cmap("rainbow")
        cNorm = colors.Normalize(vmin=0, vmax=self.n_x)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
        fig = plt.figure(figsize=(8, 8))
        plt.subplot(1, 1, 1)
        for i in range(self.n_x):
            plt.scatter(exp_data[i, :], model_data[i, :], s=10, color=scalarMap.to_rgba(i))
        plt.xlabel('RNAseq Data')
        plt.ylabel('Model Solution')
        plt.title("Model vs Data")
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()
