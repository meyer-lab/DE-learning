""" Organizing the data into tensor format. """
from .importData import importLINCS
from .factorization import commonGenes


def tensor():
    """ find all common genes and create a tensor format data from genes, cell lines, and gene expression levels. """
    A375, gA375 = importLINCS("A375")
    A549, gA549 = importLINCS("A549")
    HA1E, gHA1E = importLINCS("HA1E")
    HT29, gHT29 = importLINCS("HT29")
    MCF7, gMCF7 = importLINCS("MCF7")
    PC3, gPC3 = importLINCS("PC3")

    ids = commonGenes([gA375, gA549, gHA1E, gHT29, gMCF7, gPC3])


