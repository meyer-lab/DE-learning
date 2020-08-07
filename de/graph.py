"""Contains functions for creating directed graph from w matrix."""
from os.path import join, dirname
import numpy as np
import pandas as pd
import networkx as nx

def load_w():
    """
    Loads w from csv file and returns dataframe with gene symbols attached to w values.
    """
    path_here = dirname(dirname(__file__))
    w = pd.read_csv(join(path_here, "de/data/w.csv"), header=None)
    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)
    w.columns = genes
    w.index = genes
    return w

def add_nodes(dir_graph, w):
    """
    Given a directed graph and w matrix, adds a node to the directed graph for each gene.
    """
    for i in range(83):
        dir_graph.add_node(i, gene=w.columns[i])
    return dir_graph

def add_edges(dir_graph, w):
    """
    Given a directed graph and w matrix, calculates a threshold for large w values. Then adds a directed edge from gene j to gene i representing the interaction with the w value as the edge's weight.
    """
    w = w.to_numpy()
    w_new = abs(w)
    w_max = np.max(w_new)
    threshold = np.mean(w_new) + 1.5 * np.std(w_new)
    for i in range(83):
        for j in range(83):
            if w_new[i, j] > threshold:
                if w[i, j] > 0:
                    dir_graph.add_edge(j, i, color="red", weight=w[i, j])
                else:
                    dir_graph.add_edge(j, i, color="blue", weight=w[i, j])
    # Remove nodes with no edges
    isolates = list(nx.isolates(dir_graph))
    dir_graph.remove_nodes_from(isolates)
    return dir_graph, threshold, w_max
