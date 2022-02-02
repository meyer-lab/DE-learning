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
    Tensor[:, :n, 0] = A37[:, ids[0]]
    A54 = A549[ids[1], :]
    Tensor[:, :n, 1] = A54[:, ids[1]]
    HA1 = HA1E[ids[2], :]
    Tensor[:, :n, 2] = HA1[:, ids[2]]
    HT2 = HT29[ids[3], :]
    Tensor[:, :n, 3] = HT2[:, ids[3]]
    MCF = MCF7[ids[4], :]
    Tensor[:, :n, 4] = MCF[:, ids[4]]
    PC = PC3[ids[5], :]
    Tensor[:, :n, 5] = PC[:, ids[5]]

    # controls
    Tensor[:, n, 0] = A375[ids[0], -1]
    Tensor[:, n, 1] = A549[ids[1], -1]
    Tensor[:, n, 2] = HA1E[ids[2], -1]
    Tensor[:, n, 3] = HT29[ids[3], -1]
    Tensor[:, n, 4] = MCF7[ids[4], -1]
    Tensor[:, n, 5] = PC3[ids[5], -1]

    # zscore each gene
    for i in range(Tensor.shape[2]):
        Tensor[:, :, i] = zscore(Tensor[:, :, i], axis=0)

    # assert the genes are the same among cell line1 and 2
    assert(np.all(np.array(gA375)[ids[0]] == np.array(gA549)[ids[1]]))
    gene_names = np.array(gA375)[ids[0]]

    cellLines = ["A375", "A549", "HA1E", "HT29", "MCF7", "PC3"]

    return Tensor, gene_names, cellLines

