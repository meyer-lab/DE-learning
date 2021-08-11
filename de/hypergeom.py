"""Contains functions to use a hypergeometric distribution to compare the GRNdb network with ours."""
from scipy.stats import hypergeom
import networkx as nx

def setvars(G, G_GRNdb):
    """
    Takes in our network graph and the GRNdb network graph, returns the variables needed for the hypergeometric distribution.
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
    """
    k, M, n, N = varlist[0], varlist[1], varlist[2], varlist[3]
    prob = hypergeom.pmf(k, M, n, N, loc=0)
    return prob

def CDF(varlist):
    """
    Takes in a list of variables [k, M, n, N] and computes the result of their cumulative distribution function.
    """
    k, M, n, N = varlist[0], varlist[1], varlist[2], varlist[3]
    prob = hypergeom.cdf(k, M, n, N, loc=0)
    return prob
