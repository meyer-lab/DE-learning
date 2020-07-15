'''
Basic Model simulation
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from .ODE import solver, sim_KO, julia_solver, julia_sol_matrix

class Model():
    '''
    The class for basic model simulation
    '''
    def __init__(self, n_x=83, N=84):
        self.n_x = n_x #The number of components/genes involved
        self.N = N #The number of knockout conditions

    def graph(self, t, sol, save_path):
        '''
        Function receives simulation of ODE, condition name, time series, model solution, and path at which to save image. 
        Generates graphs of the solution overtime.
        '''
        fig = plt.figure(figsize=(100, 300.0))
        label = np.loadtxt('./data/node_Index.csv', delimiter=',', dtype=str)
        for i in range(self.n_x):
            axes = fig.add_subplot(math.ceil(self.n_x/5), 5, i+1)
            axes.plot(t, sol[:, i], 'r')
            axes.set_title(label[i])
        plt.xlabel('t')
        plt.ylabel('Expression Level')  
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def py_sim(self, params, N_t, tp):
        '''Function receives simulation of ODE, 1D array of optimizable parameters, time series, initial values and simulation type.
        Runs the ODE model and returns solution overtime(negative control)/ at last timepoints(different KO conditions).'''
        if tp == 'overtime':
            sol = solver(N_t, params, tp)
        elif tp == 'endpoint':
            sol = np.zeros((self.n_x,self.N))
            for i in range(self.N-1):
                sol[:, i] = sim_KO(params, i, N_t, tp)
            sol[:, (self.N-1)] = solver(N_t, params, tp)
        return sol

    def random_params(self):
        '''
        Function receives simulation of ODE and randomly initialize the parameters based on the number of components.
        Returns 1D array of parameters p = [eps (n_x), w (n_x ** 2), alpha (n_x)]
        '''
        eps = np.abs(np.random.normal(1, 1.0, size=(self.n_x)))
        W = np.random.normal(0.01, 1.0, size=(self.n_x, self.n_x))
        W_mask = (1.0 - np.diag(np.ones([self.n_x]))) #remove self-interaction
        w = W_mask*W
        alpha = np.abs(np.random.normal(2, 1.0, size=(self.n_x)))
        # Combine all parameters into p
        p = np.concatenate([eps, w.flatten(), alpha])
        return p

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
