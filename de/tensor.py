""" Organizing the data into tensor format. """
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
    A375 = A375[ids[0][0:-1], :]
    A375 = A375[:, ids[0]]
    A549 = A549[ids[1][0:-1], :]
    A549 = A549[:, ids[1]]
    HA1E = HA1E[ids[2][0:-1], :]
    HA1E = HA1E[:, ids[2]]
    HT29 = HT29[ids[3][0:-1], :]
    HT29 = HT29[:, ids[3]]
    MCF7 = MCF7[ids[4][0:-1], :]
    MCF7 = MCF7[:, ids[4]]
    PC3 = PC3[ids[5][0:-1], :]
    PC3 = PC3[:, ids[5]]

    # create a tensor of gene expressions x gene perturbations x cell lines
    Tensor = np.zeros((A375.shape[0], A375.shape[1], len(ids)))
    Tensor[:, :, 1] = A375
    Tensor[:, :, 2] = A549
    Tensor[:, :, 3] = HA1E
    Tensor[:, :, 4] = HT29
    Tensor[:, :, 5] = MCF7
    Tensor[:, :, 6] = PC3
    return Tensor