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
#     w = np.transpose(w)
    w_new = abs(w)
    w_max = np.max(w_new)
    threshold = np.mean(w_new) + np.std(w_new)
    for i in range(83):
        for j in range(83):
            if w_new[i, j] > threshold:
                if w[i, j] > 0:
                    dir_graph.add_edge(j, i, color="red", weight=w_new[i, j])
                else:
                    dir_graph.add_edge(j, i, color="blue", weight=w_new[i, j])
    # Remove nodes with no edges
    isolates = list(nx.isolates(dir_graph))
    dir_graph.remove_nodes_from(isolates)
    return dir_graph, threshold, w_max

def pagerank(dir_graph, pos, alpha = 0.85):
    """
    Given a directed graph and position type, calculate the pagerank for each node. Then return a directed edge with adjusted nodesize based on pagerank.
    """
    pagerank = nx.pagerank(dir_graph, alpha, weight = (dir_graph[u][v]['weight'] for u,v in dir_graph.edges()))
    nx.set_node_attributes(dir_graph, name = 'pagerank', values=pagerank)
    nodesize = [v['pagerank']*20000 for u,v in dir_graph.nodes(data=True)]
    
    #draw the nodes
    nx.draw_networkx_nodes(dir_graph, pos, node_size=nodesize)
    return dir_graph

def adjustment(dir_graph, threshold, w_max, pos):
    """
    Given a directed graph, threshold for w and w_max, calculate edges color and thickness. Then return a directed edge.
    """
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u,v in edges]
    thickness = [np.exp((dir_graph[u][v]['weight'] - threshold) / (w_max - threshold)) for u,v in edges]
    
    #draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, width=thickness, edge_color=colors)
    return dir_graph
