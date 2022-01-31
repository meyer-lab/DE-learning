import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from ..tensor import factorize

def cluster_map():
    """ Get a list of the axis objects and create a figure. """

    tFac, _, gene_names, _ = factorize(num_comp=6)
    genes = pd.DataFrame(tFac.factors[0], index=gene_names, columns=["comp1", "comp2", "comp3", "comp4", "comp5", "comp6"])
    decreased_genes = genes.loc[((-0.1 >= genes).any(1) | (genes >= 0.1).any(1))]
    g = sns.clustermap(decreased_genes, cmap="bwr", method="centroid", center=0, figsize=(14, 20))
    plt.savefig("output/clustergram_genes.svg")

    perturbation_list = list(gene_names).append("control")
    perturbations = pd.DataFrame(tFac.factors[1], index=perturbation_list, columns=["comp1", "comp2", "comp3", "comp4", "comp5", "comp6"])
    decreased_perturbations = perturbations.loc[((-0.1 >= perturbations).any(1) | (perturbations >= 0.1).any(1))]
    g = sns.clustermap(decreased_perturbations, cmap="bwr", method="centroid", center=0, figsize=(14, 20))
    plt.savefig("output/clustergram_perturbations.svg")