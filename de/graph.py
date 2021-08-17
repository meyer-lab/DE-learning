""" Contains functions for creating directed graph from w matrix. """
from os.path import join, dirname
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from .importData import ImportMelanoma
from .factorization import factorizeEstimate


def load_w():
    """
    Loads w from csv file and returns dataframe with gene symbols attached to w values.
    
    :output w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type: DataFrame
    """
    path_here = dirname(dirname(__file__))

    data = ImportMelanoma()
    w, _ = factorizeEstimate(data)
    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)

    return pd.DataFrame(w, columns=genes, index=genes)


def normalize(w):
    """
    Given w matrix, then return normalized w matrix according to gene expression under control conditions.

    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: DataFrame
    :output w: A normalized matrix
    :type w: DataFrame
    """
    control = ImportMelanoma()[:, -1]
    for i in range(len(control)):
        w.iloc[:, i] = w.iloc[:, i] * control[i]
    return w


def remove(w):
    """
    Removes POLR2A and genes whose expression level equals zero under control condition from w matrix.

    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: DataFrame
    :output w: An edited matrix without POLR2A and negatively expressed genes
    :type w: Array
    """
    w = w.drop(["POLR2A"], axis=0)
    w = w.drop(["POLR2A"], axis=1)
    m = w.columns
    w_2 = w.to_numpy()
    for i in range(len(m)):
        if w_2[0, i] == 0:
            w = w.drop([m[i]], axis=0)
            w = w.drop([m[i]], axis=1)
    return w


def pagerank(w, num_iterations: int = 100, d: float = 0.85):
    """
    Given an adjecency matrix, calculate the pagerank value.
    Notice: All the elements in w should be no less than zeros; Also, the elements of each column should sum up to 1.
    
    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: Array
    :output v: pagerank value representing the likelihood of ending up at each node from any other node
    :type v: Float
    """
    w = np.absolute(w)  # PageRank only works with unsigned networks, so we'll take the absolute value.
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

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: Array
    :param w_abs: A matrix of absolute values representing perturbation interactions with genes as columns and indices as gene names
    :type w_abs: Array
    :output dir_graph: An edited directed graph with genes added as nodes
    :type dir_graph: DiGraph
    """
    w_abs = np.copy(w_abs)
    v = pagerank(w_abs)
    for i in range(len(v)):
        dir_graph.add_node(i, gene=w.columns[i], pagerank=v[i])
    return dir_graph


def add_edges(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, calculates a threshold for large w values. Then adds a directed edge from gene j to gene i representing the interaction with the w value as the edge's weight.
    
    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: Array
    :param w_abs: A matrix of absolute values representing perturbation interactions with genes as columns and indices as gene names
    :type w_abs: Array
    :output dir_graph: An edited directed graph with edges added using w values as edge weight
    :type dir_graph: DiGraph
    """
    w = w.to_numpy()
    threshold = np.mean(w_abs) + 1.4 * np.std(w_abs)  # lower threshold in order to find more possible loops
    for i in range(w.shape[1]):
        for j in range(w.shape[1]):
            if w_abs[i, j] > threshold:
                if w[i, j] > 0:
                    dir_graph.add_edge(j, i, color="red", weight=w[i, j])
                else:
                    dir_graph.add_edge(j, i, color="blue", weight=w[i, j])
    return dir_graph


def remove_isolates(dir_graph):
    """
    Given a directed graph, remove nodes with no edges.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :output dir_graph: An edited directed graph without isolated nodes
    :type dir_graph: DiGraph
    """
    isolates = list(nx.isolates(dir_graph))
    dir_graph.remove_nodes_from(isolates)

    return dir_graph


def set_nodes(dir_graph, pos, ax):
    """
    Given a directed graph and pos, then draw the corresponding node based on pagerank value, color coded by gene type.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :output dir_graph: An edited directed graph with nodes of specified size and color
    :type dir_graph: DiGraph
    """
    nodes = dir_graph.nodes()
    nodesize = [dir_graph.nodes[u]["pagerank"] * 260000 for u in nodes]

    full_resistant_list = ["JUN", "BRD2", "STK11", "PKN2", "NFAT5", "KMT2D", "ADCK3", "FOSL1", "CSK", "BRD8", "CBFB", "TADA2B", "DSTYK", "JUNB", "LATS2", "FEZF2", "MITF", "RUNX3", "SUV420H1", "SOX10", "DOT1L", "PRKRIR"]
    pre_resistant_list = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A", "EGFR", "EP300", "SRF", "PRKAA1", "GATA4", "MYBL1", "MTF1"]
    unknown = []
    #color nodes based on pre/resistance
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


def set_edges(dir_graph, w_abs, w_max, pos, ax):
    """
    Given a directed graph, w_new and w_max, calculate edges color and thickness. Then draw the corresponding edge.
    
    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :param w_abs: A matrix of absolute values representing perturbation interactions with genes as columns and indices as gene names
    :type w_abs: Array
    :param w_max: The maximum value of all absolute values in w 
    :type w_abs: NDArray
    :output dir_graph: An edited directed graph with edges of specified color and thickness
    :type dir_graph: DiGraph
    """
    threshold = np.mean(w_abs) + 1.4 * np.std(w_abs)
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u, v in edges]
    thickness = [np.exp((np.abs(dir_graph[u][v]["weight"]) - threshold) / (w_max - threshold)) for u, v in edges]

    # to use this as alpha, normalize between 0.2, 1.0
    normalized_thickness = ((thickness - np.min(thickness)) / np.ptp(thickness)) * 0.8 + 0.2
    # draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, width=thickness, edge_color=colors, arrowsize=65, ax=ax, alpha=normalized_thickness)
    return dir_graph


def set_labels(dir_graph, pos, ax):
    """
    Given a directed graph and pos, then draw the corresponding label based on index.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :output dir_graph: An edited directed graph with nodes labeled with gene names
    :type dir_graph: DiGraph
    """
    labels = nx.get_node_attributes(dir_graph, "gene")

    # draw the labels
    nx.draw_networkx_labels(dir_graph, pos, labels=labels, font_size=48, ax=ax)
    return dir_graph

def make_legend(dir_graph, ax):
    """ Creates a legend for node and edge colors in Network.

    :param dir_graph: A directed graph of gene interactions
    :type dir_graph: DiGraph
    :output dir_graph: An edited directed graph with a legend
    :type dir_graph: DiGraph
    """
    purple_patch = mpatches.Patch(color="darkorchid", label="Pre-resistant")
    green_patch = mpatches.Patch(color="mediumturquoise", label="Resistant")
    grey_patch = mpatches.Patch(color="grey", label="Undetermined")
    blue_line = mlines.Line2D([], [], color="blue", label="Inhibition")
    red_line = mlines.Line2D([], [], color="red", label="Activation")
    ax.legend(handles=[purple_patch, green_patch, grey_patch, red_line, blue_line], prop=dict(size=50))
    return dir_graph

def Network(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, then draw the corresponding Networkx graph.

    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: Array
    :param w_abs: A matrix of absolute values representing perturbation interactions with genes as columns and indices as gene names
    :type w_abs: Array
    :param w_max: The maximum value of all absolute values in w 
    :type w_abs: NDArray
    :output G: Networkx weighted directed graph
    :type G: DiGraph
    """
    G = nx.DiGraph()
    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)

    pos = nx.spring_layout(G, k=0.2)

    # draw the nodes, edges and labels
    set_nodes(G, pos, ax)
    set_edges(G, w_abs, w_max, pos, ax)
    set_labels(G, pos, ax)
    make_legend(G, ax)

    return G


def bar_graph(w, color, ax, label):
    """
    Given w, color, ax and label, then draw the corresponding bar graph based on pagerank value.

    :param w: A matrix representing perturbation interactions with genes as columns and indices as gene names
    :type w: Array
    :param color: color
    :type color: Any
    :param ax: Axis
    :type ax: Any
    :param label: Label of nodes; gene names
    :type label: String
    """

    w_new = w.to_numpy()
    v = pagerank(w_new)
    v_new = pd.DataFrame(v)
    v_new.index = w.columns
    v_new.columns = [label]
    v_new.sort_values(by=label, inplace=True, ascending=False)
    v_new[0:20].plot.bar(color=color, ax=ax)


def loop():
    """
    Return positive-feedback loops involved in network. 

    :output positive: List of positive-feedback loops
    :type positive: List
    :output G_1: Copy of network graph with positive-feedback loops removed
    :type G_1: DiGraph
    """
    w = load_w()
    w = remove(w)
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)

    G = Network(w, w_abs, w_max, ax=None)
    G_1 = G.copy()
    m = list(nx.simple_cycles(G_1))
    positive = []

    # remove self-interacting loop
    for l in m:
        if len(l) == 1:
            G_1.remove_edges_from([(l[0], l[0])])
    m_new = list(nx.simple_cycles(G_1))

    for i in m_new:
        product = 1
        for j in range(len(i)):
            if (j + 1) < len(i):
                product *= G_1[i[j]][i[j + 1]]["weight"]
            else:
                product *= G_1[i[j]][i[0]]["weight"]
        if product > 0:
            positive.append(i)
    return positive, G_1


def loop_figure(loop, G_1):
    """
    Given certain loop number(eg.[37, 74, 26, 60]) and network graph, return loop graph.

    :parameter loop: List of loop numbers
    :type loop: List
    :parameter G_1: Copy of network graph with positive-feedback loops removed
    :type G_1: DiGraph
    """
    w = load_w()
    w = remove(w)
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)

    edge = []
    node = []

    for i, loopi in enumerate(loop):
        node.append((loopi, G_1.nodes[loopi]))

    for j, loopj in enumerate(loop):
        if (j + 1) < len(loop):
            edge.append((loopj, loop[j + 1], G_1[loopj][loop[j + 1]]))
        else:
            edge.append((loopj, loop[0], G_1[loopj][loop[0]]))

    G_test = nx.DiGraph()
    G_test.add_nodes_from(node)
    G_test.add_edges_from(edge)
    pos = nx.circular_layout(G_test)
    set_nodes(G_test, pos, ax=None)
    set_edges(G_test, w_abs, w_max, pos, ax=None)
    set_labels(G_test, pos, ax=None)
