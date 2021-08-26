"""Contains functions to use a hypergeometric distribution to compare the GRNdb network with ours."""
from scipy.stats import hypergeom
import networkx as nx
import numpy as np

def setvars(G, G_GRNdb):
    """
    Takes in our w network graph and the GRNdb network graph, and returns the variables (k, M, n and N) for the hypergeometric distribution.
    
    :param G: A Networkx weighted directed graph to represent all interactions in w
    :type G: DiGraph
    :param G_GRNdb: A Networkx weighted directed graoh to represent all relevant interactions in GRNdb
    :type G_GRNdb: DiGraph
    :output [k, M, n, N]: A list of the variables needed to compute hypergeometric distribution functions
    k is the number of interactions overlapping in the w and GRNdb networks
    M is the number of total positions in the w matrix
    n is the number of total significant w entries, regardless of whether or not they overlap with GRNdb
    N is the number of relevant GRNdb entries
    :type [k, M, n, N]: list
    """
    # M = population size - total positions in w (# genes squared)
    # N = # draws - # database entries I found
    # n = # total significant w entries, regardless of whether or not they overlap with GRNdb
    M = 83**2
    N = G_GRNdb.number_of_edges()
    n = G.number_of_edges()
    
    # k = # interactions overlapping our and GRNdb networks
    # create dictionaries for each graph with key : value = node index : gene id
    G_nodes = dict(G.nodes(data='gene'))
    G_GRNdb_nodes = dict(G_GRNdb.nodes(data='gene'))
    
    # iterate through edges of G, get the gene ids from the indices, fill edge array with gene ids
    G_edge_ids = []
    for edge in list(G.edges()):
        from_gene = G_nodes.get(edge[0])
        to_gene = G_nodes.get(edge[1])
        G_edge_ids.append((from_gene, to_gene))

    # iterate through edges of G_GRNdb, get the gene ids from the indices, fill edge array with gene ids
    G_GRNdb_edge_ids = []
    for eddge in list(G_GRNdb.edges()):
        fromm_gene = G_GRNdb_nodes.get(eddge[0])
        too_gene = G_GRNdb_nodes.get(eddge[1])
        G_GRNdb_edge_ids.append((fromm_gene, too_gene))

    # create edge array to contain matching edges
    matches = []
    for pair in G_edge_ids:
        if pair in G_GRNdb_edge_ids:
            matches.append(pair)
    k = len(matches)
    
    return [k, M, n, N]

def PMF(varlist):
    """
    Takes in a list of variables [k, M, n, N] and computes the result of their probability mass function.
    
    :param varlist: A list of variables needed to compute the probability mass function
    :type varlist: list
    :output [x, pmf_edges]: Variables for plotting the PMF
    x represents the different values that k could take on in our scenario
    pmf_edges represents the probability that k = x due to random chance, for each value of x
    :type [x, pmf_edges]: list
    """
    M, n, N = varlist[1], varlist[2], varlist[3]
    rv = hypergeom(M, n, N)
    x = np.arange(0, 50)
    pmf_edges = rv.pmf(x)
    return [x, pmf_edges]

def CDF(varlist):
    """
    Takes in a list of variables [k, M, n, N] and computes the result of their cumulative distribution function.
    
    :param varlist: A list of variables needed to compute the cumulative distribution function
    :type varlist: list
    :output [x, cdf_edges]: Variables for plotting the CDF
    x represents the different values that k could take on in our scenario
    cdf_edges represents the probability that k <= x due to random chance, for each value of x
    :type [x, cdf_edges]: list
    """
    M, n, N = varlist[1], varlist[2], varlist[3]
    rv = hypergeom(M, n, N)
    x = np.arange(0, 50)
    cdf_edges = rv.cdf(x)
    return [x, cdf_edges]
