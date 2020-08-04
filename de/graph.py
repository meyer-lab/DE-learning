"""Contains functions for creating directed graph from w matrix."""
from os.path import join, dirname
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
    threshold = np.mean(w) + 2 * np.std(w)
    for i in range(83):
        for j in range(83):
            if w[i, j] > threshold:
                dir_graph.add_edge(j, i, weight=w[i, j])
    # Remove nodes with no edges
    dir_graph.remove_nodes_from(nx.isolates(dir_graph))
    return dir_graph

def make_graph(dir_graph):
    """
    Forms figure from directed graph with nodes labelled as genes.
    """
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    labels = nx.get_node_attributes(dir_graph, 'gene')
    nx.draw_networkx(dir_graph,labels=labels, node_size=200, font_size=8)
