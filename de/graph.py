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

def pagerank(w, num_iterations: int = 100, d: float = 0.85):
    """
    Given an adjecency matrix, calculate the pagerank value.
    Notice: All the elements in w should be no less than zeros; Also, the elements of each column should sum up to 1.
    """
    w = np.absolute(w) # PageRank only works with unsigned networks, so we'll take the absolute value.
    N = w.shape[1]
    for i in range(N):
        w[:, i] /= sum(w[:, i])
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    w_hat = (d * w + (1 - d) / N)
    for i in range(num_iterations):
        v = w_hat @ v
    return v

def add_nodes(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, adds a node to the directed graph for each gene.
    """
    v = pagerank(w_abs)
    for i in range(83):
        dir_graph.add_node(i, gene=w.columns[i], pagerank=v[i])
    return dir_graph

def add_edges(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, calculates a threshold for large w values. Then adds a directed edge from gene j to gene i representing the interaction with the w value as the edge's weight.
    """
    w = w.to_numpy()
    threshold = np.mean(w_abs) + 1.5 * np.std(w_abs)
    for i in range(83):
        for j in range(83):
            if w_abs[i, j] > threshold:
                if w[i, j] > 0:
                    dir_graph.add_edge(j, i, color="red", weight=w_abs[i, j])
                else:
                    dir_graph.add_edge(j, i, color="blue", weight=w_abs[i, j])
    return dir_graph

def remove_isolates(dir_graph):
    """
    Given a directed graph, then remove nodes with no edges.
    """
    isolates = list(nx.isolates(dir_graph))
    dir_graph.remove_nodes_from(isolates)

    return dir_graph

def set_nodes(dir_graph, pos):
    """
    Given a directed graph and pos, then draw the corresponding node based on pagerank value.
    """
    nodes = dir_graph.nodes()
    nodesize = [dir_graph.nodes[u]["pagerank"]*20000 for u in nodes]

    #draw the nodes
    nx.draw_networkx_nodes(dir_graph, pos, node_size=nodesize)
    return dir_graph

def set_edges(dir_graph, w_abs, w_max, pos):
    """
    Given a directed graph, w_new and w_max, calculate edges color and thickness. Then draw the corresponding edge.
    """
    threshold = np.mean(w_abs) + 1.5 * np.std(w_abs)
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u, v in edges]
    thickness = [np.exp((dir_graph[u][v]["weight"] - threshold) / (w_max - threshold)) for u, v in edges]

    #draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, width=thickness, edge_color=colors)
    return dir_graph
        
def set_labels(dir_graph, pos):
    """
    Given a directed graph and pos, then draw the corresponding label based on index.
    """
    labels = nx.get_node_attributes(dir_graph, "gene")
         
    #draw the labels
    nx.draw_networkx_labels(dir_graph, pos, labels=labels, font_size=8)
    return dir_graph
