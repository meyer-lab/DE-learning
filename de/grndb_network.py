"""Contains functions for creating w matrix and directed graph for GRNdb dataset."""
from os.path import join, dirname
import networkx as nx
import pandas as pd
import numpy as np
from .graph import remove_isolates, set_nodes, set_labels

def load_w_GRNdb():
    """
    Loads node labels and creates w matrix as a dataframe.
    """
    path_here = dirname(dirname(__file__))

    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)

    # create matrix of 0's
    w_df = pd.DataFrame(columns=genes, index=genes)
    w_df = w_df.fillna(0)

    interactions = pd.read_csv("de/data/grndb/SKCM_all_edited.csv")

    # fill in 1's where there is a TF>gene interaction
    for row in interactions.index:
        TF = interactions['TF'][row]
        TG = interactions['gene'][row]
        w_df.at[TG, TF] = 1

    return w_df

def pagerank_GRNdb(w, num_iterations: int = 100, d: float = 0.85):
    """
    Given an adjacency matrix, calculate the pagerank value.
    """
    w = np.absolute(w)  # PageRank only works with unsigned networks, so we'll take the absolute value.
    N = w.shape[1]
    for i in range(N):
        if sum(w.iloc[:, i]) != 0: # avoid dividing by zero
            w.iloc[:, i] /= sum(w.iloc[:, i])
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    w_hat = (d * w + (1 - d) / N)
    for i in range(num_iterations):
        v = w_hat @ v
    return v

def add_nodes_GRNdb(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, adds a node to the directed graph for each gene.
    """
    w_abs = np.copy(w_abs)
    v = pagerank_GRNdb(w_abs)
    for i in range(len(v)):
        dir_graph.add_node(i, gene=w.columns[i], pagerank=v[i])
    return dir_graph

def add_edges_GRNdb(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, adds an unweighted, directed edge from gene j to gene i representing an interaction, if it exists.
    """
    w = w.to_numpy()
    for i in range(w.shape[1]):
        for j in range(w.shape[1]):
            if w[i, j] > 0:
                dir_graph.add_edge(j, i, color="green")
    return dir_graph

def set_edges_GRNdb(dir_graph, pos, ax):
    """
    Given a directed graph, draws singly colored, unweighted, directed edges to represent interactions.
    """
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u, v in edges]

    # draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, edge_color=colors, arrowsize=65, ax=ax)
    return dir_graph

def Network_GRNdb(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, draw the corresponding Networkx graph.
    """
    G = nx.DiGraph()

    # add nodes and edges
    add_nodes_GRNdb(G, w, w_abs)
    add_edges_GRNdb(G, w, w_abs)

    # remove unconnected nodes
    remove_isolates(G)

    pos = nx.spring_layout(G, k=0.2)

    # draw the nodes, edges, labels and legend
    set_nodes(G, pos, ax)
    set_edges_GRNdb(G, pos, ax)
    set_labels(G, pos, ax)
    # make_legend(G, ax)

    return G

