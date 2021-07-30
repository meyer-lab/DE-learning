"""
This creates Figure 2: w Network Graph
"""
from networkx.algorithms.cluster import _weighted_triangles_and_degree_iter
import numpy as np
import networkx as nx
import random 
import bellmanford as bf
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
    w_full, w_pre, w_rand = cluster_dist()
    ax[2].hist([w_full, w_pre, w_rand], alpha=0.7, label=["Full R", "Pre R", "Random"])
    ax[2].legend()
    ax[2].set_xlabel("Node distance")
    ax[2].set_ylabel("Frequency")
    ax[2].set_title("Network distance distributions")
    max_dist = np.max(np.concatenate([w_full, w_pre, w_rand])) #takes maximum of distance values
    ax[2].set_xticks((np.linspace(1.0,max_dist, max_dist)))
    # create upstream bar graph
    bar_graph(w_trans, "orange", ax[3], "upstream")
    # set title for the graph
    ax[3].set_title("Bar Graph (upstream)")
    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f

def cluster_dist():
    G = nx.Graph()
    w = load_w()
    w = normalize(w)
    w = remove(w)
    w_abs = np.absolute(w.to_numpy())
    
    # add nodes and edges
    add_nodes(G, w, w_abs)
    add_edges(G, w, w_abs)
    remove_isolates(G)

    w_full = []
    w_pre = []
    w_rand = []
    
    full = [24, 1, 65, 45, 40, 29, 0, 17, 9, 2, 4, 67, 13, 25, 30, 37, 57, 66, 60, 12, 50]
    pre = [34, 33, 43, 27, 15, 16, 64, 48, 18, 39, 38]
    for _ in range(70):
        temp1 = random.sample(full, 2)
        w_full.append(bf.bellman_ford(G, source=temp1[0], target=temp1[1], weight="length")[0]) # the first output of the function is the path length
        temp2 = random.sample(pre, 2)
        w_pre.append(bf.bellman_ford(G, source=temp2[0], target=temp2[1], weight="length")[0])
        temp3 = np.concatenate([np.random.choice(full,1), np.random.choice(pre, 1)])
        w_rand.append(bf.bellman_ford(G, source=temp3[0], target=temp3[1], weight="length")[0]) 

    return w_full, w_pre, w_rand #np.mean(w_full), np.mean(w_pre), np.mean(w_rand)
