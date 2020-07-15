'''
Basic Model simulation
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from .ODE import solver, julia_solver, julia_sol_matrix

def graph(cond, t, sol, save_path):
    '''
    Function receives condition name, time series, model solution, and path at which to save image. 
    Generates graphs of the solution over time.
    '''
    fig = plt.figure(figsize=(100, 300.0))
    for i in range(83):
        axes = fig.add_subplot(17, 5, i+1)
        axes.plot(t, sol[:, i], 'r')
        axes.set_title(cond)
    plt.xlabel('t')
    plt.ylabel('Expression Level')   # x(t)/x(0)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()

def random_params():
    '''
    Function receives simulation of ODE and randomly initialize the parameters based on the number of components.
    Returns 1D array of parameters p = [eps (n_x), w (n_x ** 2), alpha (n_x)]
    '''
    eps = np.abs(np.random.normal(1, 1.0, size=(83)))
    W = np.random.normal(0.01, 1.0, size=(83, 83))
    W_mask = (1.0 - np.diag(np.ones([83]))) #remove self-interaction
    w = W_mask*W
    alpha = np.abs(np.random.normal(2, 1.0, size=(83)))
    # Combine all parameters into p
    p = np.concatenate([eps, w.flatten(), alpha])
    return p

def scatterplot(exp_data, model_data, save_path):
    '''
    Function receives matrix of RNAseq data, matrix of model solution, and the save path for image.
    Creates scatterplot of model data vs experimental data
    '''
    rainbow = plt.get_cmap("rainbow")
    cNorm = colors.Normalize(vmin=0, vmax=83)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    for i in range(83):
        plt.scatter(exp_data[i, :], model_data[i, :], s=10, color=scalarMap.to_rgba(i))
    plt.xlabel('RNAseq Data')
    plt.ylabel('Model Solution')
    plt.title("Model vs Data")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()
