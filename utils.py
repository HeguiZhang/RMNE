
import os
import networkx as nx
import numpy as np
import pickle

def read_graphs(current_path, n_views):
    """
        Read graph/network data for each view from an adjlist (from networkx package)
    :param current_path: path for graph data
    :param n_views: number of views
    :return: A list of graphs
    """
    entries = os.listdir(current_path)
    G = []
    if len(entries) != n_views:
        print("WARNING: Number of networks in the folder is not equal to number of views setting.")
    for n_net in range(n_views):
        G.append(nx.read_adjlist(current_path + entries[n_net]))
        print("Network ", (n_net + 1), ": ", entries[n_net])
    return G

def relabel_G(graph, node2idx):
    G = []
    for g in graph:
        G.append(nx.relabel_nodes(g, node2idx))
    r_G = []

    N1 = len(G[0]) + 1
    N2 = len(G[1]) + len(G[0]) + 2
    r2_mapping = {node: str(int(node) + N1) for node in G[1].nodes}
    r3_mapping = {node: str(int(node) + N2) for node in G[2].nodes}
    r_G.append(nx.relabel_nodes(G[0], {node: str(node) for node in G[0].nodes}))
    r_G.append(nx.relabel_nodes(G[1], r2_mapping))
    r_G.append(nx.relabel_nodes(G[2], r3_mapping))

    return r_G

def read_word2vec_pairs(current_path, nviews):
    """
    :param current_path: path for two files, one keeps only the node indices, the other keeps only the neighbor node
    indices of already generated pairs (node,neighbor), i.e, node indices and neighbor indices are kept separately.
    method "construct_word2vec_pairs" can be used to obtain these files.
    :E.g.:

      for pairs (9,2) (4,5) (8,6) one file keeps 9 4 8 the other file keeps 2 5 6.

    :param nviews: number of views
    :return: Two lists for all views, each list keeps the node indices of node pairs (node, neigh).
    nodes_idx_nets for node, neigh_idx_nets for neighbor
    """

    nodes_idx_nets = []
    neigh_idx_nets = []
    node_role_nets = []
    view1 = []
    view2 = []
    view3 = []
    for n_net in range(nviews):
        neigh_idx_nets.append(np.loadtxt(current_path + "/neighidxPairs_" + str(n_net + 1) + ".txt"))
        nodes_idx_nets.append(np.loadtxt(current_path + "/nodesidxPairs_" + str(n_net + 1) + ".txt"))
        view1.append(np.loadtxt(current_path + "nodesidxPairs_" + '1' + '_' + str(n_net + 1) + ".txt"))
        view2.append(np.loadtxt(current_path + "nodesidxPairs_" + '2' + '_' + str(n_net + 1) + ".txt"))
        view3.append(np.loadtxt(current_path + "nodesidxPairs_" + '3' + '_' + str(n_net + 1) + ".txt"))

    node_role_nets.append(view1)
    node_role_nets.append(view2)
    node_role_nets.append(view3)
    return nodes_idx_nets, neigh_idx_nets, node_role_nets

def degree_nodes_common_nodes(G, common_nodes, node2idx):
    """
    Assigns scores for negative sampling distribution
    """
    degrees_idx = dict((node2idx[v], 0) for v in common_nodes)
    multinomial_nodesidx = []
    for node in common_nodes:
        degrees_idx[node2idx[node]] = sum([G[n].degree(node) for n in range(len(G))])
    for node in common_nodes:
        multinomial_nodesidx.append(degrees_idx[node2idx[node]] ** (0.75))

    return multinomial_nodesidx

def save_data(data_path,data):
    with open (data_path,'wb') as f:
        pickle.dump(data,f)
    return
def read_data(G_path):
    with open(G_path, 'rb') as f:
        data = pickle.load(f)
    return data
