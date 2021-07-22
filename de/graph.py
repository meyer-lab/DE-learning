"""Contains functions for creating directed graph from w matrix."""
from os.path import join, dirname
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from .importData import ImportMelanoma, importgenes, splitnodes
from .fitting import runOptim, reshapeParams


def load_w(linear=False):
    """
    Loads w from csv file and returns dataframe with gene symbols attached to w values.
    """
    path_here = dirname(dirname(__file__))

    data = ImportMelanoma()
    ps = runOptim(data, niter=400, disp=True, linear=linear)
    w = reshapeParams(ps, data.shape[0])[0]
    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)

    return pd.DataFrame(w, columns=genes, index=genes)


def normalize(w):
    """
    Given w matrix, then return normalized w matrix according to gene expression under control conditions
    """
    control = ImportMelanoma()[:, -1]
    for i in range(len(control)):
        w.iloc[:, i] = w.iloc[:, i] * control[i]
    return w


def remove(w):
    """
    Removes POLR2A and genes whose expression level equals zero under control condition from w matrix.
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
    """
    w_abs = np.copy(w_abs)
    v = pagerank(w_abs)
    for i in range(len(v)):
        dir_graph.add_node(i, gene=w.columns[i], pagerank=v[i])
    return dir_graph


def add_edges(dir_graph, w, w_abs):
    """
    Given a directed graph and w matrix, calculates a threshold for large w values. Then adds a directed edge from gene j to gene i representing the interaction with the w value as the edge's weight.
    """
    w = w.to_numpy()
    threshold = np.mean(w_abs) + 0.2 * np.std(w_abs)  # lower threshold in order to find more possible loops
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
    Given a directed graph, then remove nodes with no edges.
    """
    isolates = list(nx.isolates(dir_graph))
    dir_graph.remove_nodes_from(isolates)

    return dir_graph


def set_nodes(dir_graph, pos, ax):
    """
    Given a directed graph and pos, then draw the corresponding node based on pagerank value.
    """
    nodes = dir_graph.nodes()
    nodesize = [dir_graph.nodes[u]["pagerank"] * 260000 for u in nodes]

def set_nodes(dir_graph, pos, ax):
    """
    Given a directed graph and pos, then draw the corresponding node based on pagerank value.
    """
    nodes = dir_graph.nodes()
    nodesize = [dir_graph.nodes[u]["pagerank"] * 260000 for u in nodes]

    pre_resistant_list = ["JUN", "BRD2", "STK11", "PKN2", "NFAT5", "KMT2D", "ADCK3", "FOSL1", "CSK", "BRD8", "CBFB", "TADA2B", "DSTYK", "JUNB", "LATS2", "FEZF2", "MITF", "RUNX3", "SUV420H1", "SOX10", "DOT1L", "PRKRIR", 'FEZF2', 'SOX10', 'ADCK3', 'BRD8', 'CBFB', 'CSK', 'DOT1L', 'DSTYK', 'FOSL1', 'JUN', 'JUNB', 'KMT2D', 'LATS2', 'MITF', 'NFAT5', 'PKN2', 'PRKRIR', 'RUNX3', 'STK11', 'SUV420H1'] 
    full_resistant_list = ["MAP3K1", "MAP2K7", "NSD1", "KDM1A", "EGFR", "EP300", "SRF", "PRKAA1", "GATA4", "MYBL1", "MTF1", 'EGFR', 'EP300', 'GATA4', 'KDM1A', 'MAP2K7', 'MAP3K1', 'MTF1', 'MYBL1', 'NSD1', 'PRKAA1', 'SRF']
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
    """
    threshold = np.mean(w_abs) + 0.2 * np.std(w_abs)
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
    """
    labels = nx.get_node_attributes(dir_graph, "gene")

    # draw the labels
    nx.draw_networkx_labels(dir_graph, pos, labels=labels, font_size=48, ax=ax)
    return dir_graph

def make_legend(dir_graph, ax):
    """ This creates legends for nodes and edges in Network """
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
    Return positive-feedback loops involved in network
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
    Given certain loop number(eg.[37, 74, 26, 60]) and network graph
    Return loop graph
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
