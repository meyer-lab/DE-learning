"""
This creates Figure 2: w Network Graph
"""
import numpy as np
import networkx as nx
import bellmanford as bf
from networkx.algorithms.shortest_paths.weighted import _bellman_ford, bellman_ford_path_length, single_source_dijkstra
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

    # create upstream bar graph
    bar_graph(w_trans, "orange", ax[3], "upstream")
    # set title for the graph
    ax[3].set_title("Bar Graph (upstream)")
    # Add subplot labels
    subplotLabel(ax, fntsize=50)
    return f

def cluster_dist():
    G = nx.Graph()
    full = ["JUNB", "MITF", "SOX10"] 
    pre = ["MAP3K1", "MTF1", "SRF"]

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
    
    #for _ in range(5):
        nodes_list  = np.array(G.nodes(data="gene"))
        node_id = np.where(nodes_list == full)[0][0]
        #temp1 = np.random.choice(full, 2)
    path_length, path_nodes, negative_cycle = bf.bellman_ford(G, source=37, target=25, weight="length")
        #id1 = [(np.where(nodes_list == (temp1[0]))), (np.where(nodes_list == (temp1[1])))] #types are not equal 
        #w_full.append(single_source_dijkstra(G, source=str(id1[0]), target=str(id1[1]), weight=True)) 
        #temp2 = np.random.choice(pre, 2)
        #w_pre.append(single_source_dijkstra(G, source=temp2[0], target=temp2[1], weight=True))
        #temp3 = np.concatenate([np.random.choice(full,1), np.random.choice(pre, 1)])
        #w_rand.append(single_source_dijkstra(G, source=temp3[0], target=temp3[1], weight=True))
    return(path_length, path_nodes, negative_cycle)
       