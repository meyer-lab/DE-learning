"""Contains functions to use a hypergeometric distribution to compare the GRNdb network with ours."""
from scipy.stats import hypergeom
import networkx as nx

def PMF(G, G_GRNdb):
    # M = population size - total positions in w (# genes squared)
    # N = # draws - # database entries I found
    # n = # objects in population with that characteristic - # total significant w entries, regardless of whether or not they overlap with GRNdb
    # k = # objects that we drew with that characteristic - # interactions overlapping our and GRNdb networks
    M = G.number_of_nodes()**2
    N = G_GRNdb.number_of_edges()
    n = G.number_of_edges()
    # k: need some way to match filtered edges/interactions of both graphs
    prb = hypergeom.pmf(k, M, n, N, loc=0)
    return prb

#def CDF()

