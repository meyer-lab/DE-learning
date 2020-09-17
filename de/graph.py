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
    w = pd.read_csv(join(path_here, "de/data/w_new.csv"), header=None)
    genes = np.loadtxt(join(path_here, "de/data/node_Index.csv"), dtype=str)
    w.columns = genes
    w.index = genes
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
    threshold = np.mean(w_abs) + 0.2 * np.std(w_abs) #lower threshold in order to find more possible loops
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
    nodesize = [dir_graph.nodes[u]["pagerank"]*20000 for u in nodes]

    #draw the nodes
    nx.draw_networkx_nodes(dir_graph, pos, node_size=nodesize, ax=ax)
    return dir_graph

def set_edges(dir_graph, w_abs, w_max, pos, ax):
    """
    Given a directed graph, w_new and w_max, calculate edges color and thickness. Then draw the corresponding edge.
    """
    threshold = np.mean(w_abs) + 0.2 * np.std(w_abs)
    edges = dir_graph.edges()
    colors = [dir_graph[u][v]["color"] for u, v in edges]
    thickness = [np.exp((np.abs(dir_graph[u][v]["weight"]) - threshold) / (w_max - threshold)) for u, v in edges]

    #draw the edges
    nx.draw_networkx_edges(dir_graph, pos, edgelist=edges, width=thickness, edge_color=colors, ax=ax)
    return dir_graph
        
def set_labels(dir_graph, pos, ax):
    """
    Given a directed graph and pos, then draw the corresponding label based on index.
    """
    labels = nx.get_node_attributes(dir_graph, "gene")
         
    #draw the labels
    nx.draw_networkx_labels(dir_graph, pos, labels=labels, font_size=8, ax=ax)
    return dir_graph

def Network(w, w_abs, w_max, ax):
    """
    Given w, w_abs, w_max and ax, then draw the corresponding Networkx graph.
    """
    
    
    G = nx.DiGraph()
    #add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)
    #draw the nodes, edges and labels
    pos = nx.spring_layout(G, k=8.0/G.number_of_nodes())
    set_nodes(G, pos, ax)
    set_edges(G, w_abs, w_max, pos, ax)
    set_labels(G, pos, ax)
    
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
            G_1.remove_edges_from([(l[0],l[0])])
    m_new = list(nx.simple_cycles(G_1))

    for i in m_new:
        product = 1
        for j in range(len(i)):
            if (j+1)<len(i):
                product *= G_1[i[j]][i[j+1]]["weight"]
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
    
    edge=[]
    node=[]
    
    for i in range(len(loop)):
        node.append((loop[i], G_1.nodes[loop[i]]))
        
    for j in range(len(loop)):
        if (j+1)<len(loop):
            edge.append((loop[j],loop[j+1],G_1[loop[j]][loop[j+1]]))
        else:
            edge.append((loop[j],loop[0],G_1[loop[j]][loop[0]]))
            
    G_test = nx.DiGraph()
    G_test.add_nodes_from(node)
    G_test.add_edges_from(edge)
    pos = nx.circular_layout(G_test)
    set_nodes(G_test, pos, ax = None)
    set_edges(G_test, w_abs, w_max, pos, ax = None)
    set_labels(G_test, pos, ax = None)
