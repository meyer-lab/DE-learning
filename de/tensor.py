""" Organizing the data into tensor format. """
import numpy as np
from scipy.stats import zscore
import tensorly as tl
from tensorly.decomposition import parafac

from .importData import importLINCS
from .factorization import commonGenes


def form_tensor() -> np.ndarray:
    """ find all common genes and create a tensor format data from genes, cell lines, and gene expression levels. """
    # import
    A375, gA375 = importLINCS("A375")
    A549, gA549 = importLINCS("A549")
    HA1E, gHA1E = importLINCS("HA1E")
    HT29, gHT29 = importLINCS("HT29")
    MCF7, gMCF7 = importLINCS("MCF7")
    PC3, gPC3 = importLINCS("PC3")

    # find common genes between all
    ids = commonGenes([gA375, gA549, gHA1E, gHT29, gMCF7, gPC3])
    n = ids[0].shape[0]

    Tensor = tl.zeros((n, n+1, len(ids))) # the added condition in the second dimension is the control

    # only keep common genes
    A37 = A375[ids[0], :]
    Tensor[:, :, 0] = A37[:, np.append(ids[0], [-1])]
    A54 = A549[ids[1], :]
    Tensor[:, :, 1] = A54[:, np.append(ids[1], [-1])]
    HA1 = HA1E[ids[2], :]
    Tensor[:, :, 2] = HA1[:, np.append(ids[2], [-1])]
    HT2 = HT29[ids[3], :]
    Tensor[:, :, 3] = HT2[:, np.append(ids[3], [-1])]
    MCF = MCF7[ids[4], :]
    Tensor[:, :, 4] = MCF[:, np.append(ids[4], [-1])]
    PC = PC3[ids[5], :]
    Tensor[:, :, 5] = PC[:, np.append(ids[5], [-1])]

    # assert the genes are the same among cell line1 and 2
    assert(np.all(np.array(gA375)[ids[0]] == np.array(gA549)[ids[1]]))

    gene_names = np.array(gA375)[ids[0]]
    np.append(gene_names, ['Control'])

    return zscore(Tensor, axis=1), gene_names

def factorize():
    """ Using Parafac as a tensor factorization. """
    tensor, genes = form_tensor()
    # perform parafac and CP decomposition
    r2x_parafac = []
    r2x_tucker = []
    for i in range(1, 5):
        # parafac
        fac_p = parafac(tensor, rank=i, svd="randomized_svd")
        r2x_parafac.append(1 - ((tl.norm(tl.cp_to_tensor(fac_p) - tensor) ** 2) / tl.norm(tensor) ** 2))

    print("parafac ", r2x_parafac)
