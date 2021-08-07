"""Contains functions to use a hypergeometric distribution to compare the GRNdb network with ours."""
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

def PMF():
    # M = population size - total # of possible gene-gene interactions
    # N = # draws - total # interactions/edges in our network
    # n = # objects in population with that characteristic - # interactions in GRNdb network
    # k = # objects that we drew with that characteristic - # interactions in our and GRNdb's networks
    prb = hypergeom.pmf(k, M, n, N, loc=0)



def CDF()

