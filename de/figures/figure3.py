"""
This creates Figure 2: w Network Graph
"""
from networkx.exception import NetworkXNoPath, NetworkXUnbounded
import numpy as np
import networkx as nx
import random 
from .figureCommon import subplotLabel, getSetup
from ..graph import Network, load_w, normalize, remove, bar_graph, add_nodes, add_edges, remove_isolates


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((100, 100), (2, 2))
    # load w
    w = load_w()
    w = normalize(w)
    w = remove(w)
    # Plot downstream graph
    w_abs = np.absolute(w.to_numpy())
    w_max = np.max(w_abs)
    Network(w, w_abs, w_max, ax[0])
    # set title for the graph
    ax[0].set_title("w Network Graph (downstream)")
    # create downstream bar graph
    bar_graph(w, "green", ax[1], "downstream")
    # set title for the graph
    ax[1].set_title("Bar Graph (downstream)")
    # Plot upstream graph
    w_trans = np.transpose(w)
    w_abs = np.absolute(w_trans.to_numpy())
    w_max = np.max(w_abs)

    # plot the hypothesis test distribution for full and pre-resistant and the random selection
    dist_full, dist_pre, dist_rand = cluster_dist()
    ax[2].hist([dist_full, dist_pre, dist_rand], alpha=0.7, label=["Full R", "Pre R", "Random"])
    ax[2].legend()
    ax[2].set_xlabel("Node distance")
    ax[2].set_ylabel("Frequency")
    ax[2].set_title("Network distance distributions")
    max_dist = np.max(np.concatenate([dist_full, dist_pre, dist_rand])) #takes maximum of distance values
    ax[2].set_xticks((np.linspace(0,max_dist, 10))) 
    # create upstream bar graph
    bar_graph(w_trans, "orange", ax[3], "upstream")
    # set title for the graph
    ax[3].set_title("Bar Graph (upstream)")
    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f

def cluster_dist():
    """ This function plots the distribution of distances betweek the two full- and pre-resistant clusters. """
    G = nx.Graph()
    w = load_w()
    w = normalize(w)
    w = remove(w)
    w_abs = np.absolute(w.to_numpy())

    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)

    for u,v in G.edges:
        G.edges[u, v]['weight'] = np.abs(1/G.edges[u, v]['weight'])

    dist_full = []
    dist_pre = []
    dist_rand = []
    
    full = [24, 1, 65, 45, 40, 29, 0, 17, 9, 2, 4, 67, 13, 25, 30, 37, 57, 66, 60, 12, 50]
    pre = [34, 33, 43, 27, 15, 16, 64, 48, 18, 39, 38]
    for _ in range(70):
        try:
            temp1 = random.sample(full, 2)
            dist_full.append(nx.bellman_ford_path_length(G, source=temp1[0], target=temp1[1], weight="weight")) # the first output of the function is the path length
        except NetworkXUnbounded or NetworkXNoPath:
            pass
        try:
            temp2 = random.sample(pre, 2)
            dist_pre.append(nx.bellman_ford_path_length(G, source=temp2[0], target=temp2[1], weight="weight")) 
        except NetworkXUnbounded or NetworkXNoPath:
            pass
        try:
            temp3 = np.concatenate([np.random.choice(full,1), np.random.choice(pre, 1)])
            dist_rand.append(nx.bellman_ford_path_length(G, source=temp3[0], target=temp3[1], weight="weight")) 
        except NetworkXUnbounded or NetworkXNoPath:
            pass
    return dist_full, dist_pre, dist_rand #np.mean(dist_full), np.mean(w_pre), np.mean(w_rand)
