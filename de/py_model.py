'''
Basic Model simulation
'''
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def graph(cond, t, sol, save_path=""):
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
    if save_path:
        plt.savefig(save_path)
    plt.show()


def scatterplot(exp_data, model_data, save_path=""):
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
    if save_path:
        plt.savefig(save_path)
    plt.show()
