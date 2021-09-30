"""Contains functions for creating w matrix and directed graph for GRNdb dataset."""
from os.path import join, dirname
import networkx as nx
import pandas as pd
import numpy as np
from .graph import set_labels, make_legend


def load_w_GRNdb():
    """
    Loads node labels and returns w matrix as a dataframe.

    :output w_df: A matrix representing perturbation interactions with genes as columns and gene names as indices
    :type w_df: DataFrame
    """
    path_here = dirname(dirname(__file__))

    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)

    # create matrix of 0's
    w_df = pd.DataFrame(columns=genes, index=genes)
    w_df = w_df.fillna(0)

    interactions = pd.read_csv("de/data/grndb/SKCM_all.csv")

    # fill in 1's where there is a TF>gene interaction
    for row in interactions.index:
        TF = interactions['TF'][row]
        TG = interactions['gene'][row]
        conf = interactions['Confidence'][row]
        if not TF == TG:  # exclude self-interacting loops
            if conf == "High":
                w_df.at[TG, TF] = 2
            else:
                w_df.at[TG, TF] = 1
    return w_df


def add_edges_GRNdb(dir_graph, w):
    """
    Given a directed graph and w matrix, adds a directed edge from gene j to gene i representing the interaction at
    w[i,j], if it exists. Edges are weighted based on confidence values in GRNdb; 0 marks no interaction, 1 marks a low
    confidence interaction, 2 marks a high confidence interaction. Returns the graph.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param w: A matrix representing perturbation interactions with genes as columns and gene names as indices
    :type w: DataFrame
    :output dir_graph: A directed graph of gene interactions with edges to represent interactions between genes
    :type dir_graph: DiGraph
    """
    w = w.to_numpy()
    for i in range(w.shape[1]):
        for j in range(w.shape[1]):
            if w[i, j] == 1:  # low confidence interaction
                dir_graph.add_edge(j, i, color="green", weight=w[i, j])
            if w[i, j] == 2:  # high confidence interaction
                dir_graph.add_edge(j, i, color="green", weight=w[i, j])
    return dir_graph


def set_nodes_GRNdb(dir_graph, pos, ax):
    """
    Given a directed graph, a node:position dictionary, and an axes object, draws each node on the graph.
    Node size is based on networkx pagerank value, and node color is based on the gene's resistant or pre-resistant status.
    Returns the graph.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param pos: a node:position dictionary for nodes in the directed graph
    :type pos: dict
    :param ax: an axes object to determine where to plot in the figure
    :type ax: Axes
    :output dir_graph: A directed graph of gene interactions with nodes of different sizes and colors
    :type dir_graph: DiGraph
    """
    node_pageranks = nx.pagerank(dir_graph)
    pr_vals = node_pageranks.values()
    pr_list = list(pr_vals)
    pr_list = [i * 2600 for i in pr_list]
    nodesize = np.array(pr_list)

    pre_resistant_list = ["JUN", "BRD2", "STK11", "PKN2", "NFAT5", "KMT2D", "ADCK3", "FOSL1", "CSK", "BRD8", "CBFB", "TADA2B", "DSTYK", "JUNB", "LATS2", "FEZF2", "MITF", "RUNX3", "SUV420H1", "SOX10",
                          "DOT1L", "PRKRIR", 'FEZF2', 'SOX10', 'ADCK3', 'BRD8', 'CBFB', 'CSK', 'DOT1L', 'DSTYK', 'FOSL1', 'JUN', 'JUNB', 'KMT2D', 'LATS2', 'MITF', 'NFAT5', 'PKN2', 'PRKRIR', 'RUNX3', 'STK11', 'SUV420H1']
    full_resistant_list = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A", "EGFR", "EP300", "SRF", "PRKAA1", "GATA4", "MYBL1", "MTF1", 'EGFR', 'EP300', 'GATA4', 'KDM1A', 'MAP2K7', 'MAP3K1', 'MTF1', 'MYBL1', 'NSD1', 'PRKAA1', 'SRF']
    unknown = []
    # color nodes based on pre/resistance
    color_list = []
    labels = nx.get_node_attributes(dir_graph, "gene")
    for _, gene in labels.items():
        if gene in pre_resistant_list:
            color_list.append("darkorchid")
        elif gene in full_resistant_list:
            color_list.append("mediumturquoise")
        else:
            unknown.append(gene)
            color_list.append("grey")

    # draw the nodes
    nx.draw_networkx_nodes(dir_graph, pos, ax=ax, node_size=nodesize, node_color=color_list, alpha=0.65)
    return dir_graph


def set_edges_GRNdb(dir_graph, pos, ax):
    """
    Given a directed graph, a node:position dictionary, and an axes object, draws singly colored, weighted, directed edges to represent interactions.
    Returns the graph.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param pos: a node:position dictionary for nodes in the directed graph
    :type pos: dict
    :param ax: an axes object to determine where to plot in the figure
    :type ax: Axes
    :output dir_graph: A directed graph of gene interactions with edges of different thicknesses and opacity to represent high and low confidence interactions
    :type dir_graph: DiGraph
    """
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u, v in edges]
    thickness = [dir_graph[u][v]["weight"] for u, v in edges]

    normalized_thickness = ((thickness - np.min(thickness)) / np.ptp(thickness)) * 0.3 + 0.5
    # draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, width=thickness, edge_color=colors, arrowsize=6, ax=ax, alpha=normalized_thickness)
    return dir_graph


def Network_GRNdb(w, ax):
    """
    Given a w gene-gene interaction matrix and an axes object, draw and return the corresponding Networkx graph.

    :param w: A matrix representing perturbation interactions with genes as columns and gene names as indices
    :type w: DataFrame
    :param ax: an axes object to determine where to plot in the figure
    :type ax: Axes
    :output G: Networkx weighted directed graph
    :type G: DiGraph
    """
    G = nx.DiGraph()

    # add nodes and edges
    for i in range(len(w)):
        G.add_node(i, gene=w.columns[i])
    add_edges_GRNdb(G, w)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())

    pos = nx.kamada_kawai_layout(G, dist=df.to_dict())

    # draw the nodes, edges, labels and legend
    set_nodes_GRNdb(G, pos, ax)
    set_edges_GRNdb(G, pos, ax)
    set_labels(G, pos, ax)
    make_legend(G, ax)

    # remove self-interacting loops
    m = list(nx.simple_cycles(G))
    for l in m:
        if len(l) == 1:
            G.remove_edges_from([(l[0], l[0])])

    return G
