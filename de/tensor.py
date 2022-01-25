""" Organizing the data into tensor format. """
import numpy as np
from .importData import importLINCS
from .factorization import commonGenes


def tensor() -> np.ndarray:
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
    # only keep common genes
    A37 = A375[ids[0][0:-1], :]
    a375 = A37[:, ids[0]]
    A54 = A549[ids[1][0:-1], :]
    a549 = A54[:, ids[1]]
    HA1 = HA1E[ids[2][0:-1], :]
    hA1E = HA1[:, ids[2]]
    HT2 = HT29[ids[3][0:-1], :]
    hT29 = HT2[:, ids[3]]
    MCF = MCF7[ids[4][0:-1], :]
    mCF7 = MCF[:, ids[4]]
    PC = PC3[ids[5][0:-1], :]
    pC3 = PC[:, ids[5]]

    # create a tensor of gene expressions x gene perturbations x cell lines
    Tensor = np.zeros((ids[0].shape[0]-1, ids[0].shape[0], len(ids)))
    Tensor[:, :, 0] = a375
    Tensor[:, :, 1] = a549
    Tensor[:, :, 2] = hA1E
    Tensor[:, :, 3] = hT29
    Tensor[:, :, 4] = mCF7
    Tensor[:, :, 5] = pC3
    # assert the genes are the same among cell line1 and 2
    assert(np.all(np.array(gA375)[ids[0]] == np.array(gA549)[ids[1]]))
    return Tensor, np.array(gA375)[ids[0]]
