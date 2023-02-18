"""
# reference:
# https://github.com/benedekrozemberczki/role2vec
"""
import argparse
import math
import networkx as nx
from motif_count import MotifCounterMachine
from weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import random
import pandas as pd

def join_strings(features):
    """
    Creating string labels by joining the individual quantile labels.
    """
    return {str(node): ["_".join(features[node])] for node in features} #str(node)

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
        G.append(nx.read_adjlist(current_path + entries[n_net])) #nodetype=int
        print("Network ", (n_net + 1), ": ", entries[n_net])
    return G

def create_tabular_features(args,features):

    """
    Creating tabular motifs for factorization.
    """
    binned_features = {node: [] for node in features}
    data = [ [node]+ features[node] for node in features]  #[['node' 10, 2],]
    pddata = pd.DataFrame(data)
    pddata.columns = ["id"] + ["feature_"+str(index) for index in range(len(features['1']))] #feature_1：degree
    for index in range(len(features['1'])):
        features = pddata["feature_"+str(index)].values.tolist()
        ids = pddata['id'].values.tolist()#####
        if sum(features) > 0:
            features = [math.log(feature+1) for feature in features]
            features = pd.qcut(features, args.quantiles, duplicates="drop", labels=False)
            features = dict(zip(ids, features))###
            for node in features:
                binned_features[node].append(str(int(index*args.quantiles + features[node])))
                # binned_features[node].append(str(features[node]))
    return binned_features

def create_graph_structural_features(args,graph):
        """
        Extracting structural features.
        """
        if args.features == "wl":
            features_d = {str(node): str(int(math.log(graph.degree(node)+1, args.log_base))) for node in graph.nodes()}
            ##
            # tr = nx.triangles(graph) #
            # features_d = {str(k): str(v) for k, v in tr.items()}
            ##
            machine = WeisfeilerLehmanMachine(graph, features_d, args.labeling_iterations)
            machine.do_recursions()
            features = machine.extracted_features
        elif args.features == 'motif':
            machine = MotifCounterMachine(graph, args)
            features = machine.create_string_labels()

        elif args.features == 'motif_tri':
            # features_d = {str(node): [str(int(math.log(graph.degree(node)+1, args.log_base)))] for node in graph.nodes()}
            features_d = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            tr = nx.triangles(graph)
            # features_tr = {str(node): str(int(math.log(tr[node] + 1, args.log_base))) for node in tr}
            features_tr = {str(node): str(tr[node]) for node in tr}
            for node in features_d:
                features_d[node].append(features_tr[node])
            ###string feature
            features = join_strings(features_d)
        elif args.features == 'motif_q':
            features = {str(node): [int(graph.degree(node))] for node in graph.nodes()}
            tr = nx.triangles(graph)
            features_tr = {str(node): int(tr[node]) for node in tr}
            for node in features:
                features[node].append(features_tr[node])  #{'node':[10,2]}
        else:
            #features = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            features = {str(node): [str(int(math.log(graph.degree(node)+1, args.log_base)))] for node in graph.nodes()}
        return features

def get_roles_for_one_graph(data):
    value_list = set([i[0] for i in data.values()])
    roles_nodes = {role: [] for role in value_list}
    for role in value_list:
        for node in data:
            if data[str(node)][0] == role or  data[str(node)] == role:
                roles_nodes[role].append(str(node))
    return roles_nodes,value_list

def structual_roles(args,graphs):
    graphs_roles = {}
    structure_feature_list = []
    all_roles = []
    all_roles_set = set()
    for i in range(len(graphs)):
        structure_feature = create_graph_structural_features(args,graphs[i])
        structure_feature_list.append(structure_feature)
        #print(structure_feature)
        graphs_roles[i],value_list = get_roles_for_one_graph(structure_feature)
        all_roles.append(value_list)
        all_roles_set = all_roles_set|value_list
    ###
    for i in range(len(graphs)):
        for role in all_roles_set-all_roles[i]:
            graphs_roles[i][role] = []
    return structure_feature_list, graphs_roles  #{0：{'role':['1','20'...]...},1:{},...}

def relabel_G(graph,node2idx):
    G = []
    for g in graph:
        G.append(nx.relabel_nodes(g,node2idx))
    r_G = []
    if len(G)==3:
        N1 = len(G[0]) + 1
        N2 = len(G[1]) + len(G[0]) + 2
        r2_mapping = {node: str(int(node) + N1 ) for node in G[1].nodes}
        r3_mapping = {node: str(int(node) + N2 ) for node in G[2].nodes}
        r_G.append(nx.relabel_nodes(G[0],{node:str(node) for node in G[0].nodes}))
        r_G.append(nx.relabel_nodes(G[1], r2_mapping))
        r_G.append(nx.relabel_nodes(G[2],r3_mapping))
    else:
        N1 = len(G[0]) + 1
        print('N1：')
        print(N1)
        r2_mapping = {node: str(int(node) + N1) for node in G[1].nodes()}
        r_G.append(nx.relabel_nodes(G[0], {node: str(node) for node in G[0].nodes}))
        r_G.append(nx.relabel_nodes(G[1], r2_mapping))
    return r_G

def save_node_RoleMembers(node_roleMembers, out_file):
    """
    Save node_roleMembers.
    """
    all_data = []
    for node in sorted(list(node_roleMembers.keys())):
        roles_m = node_roleMembers[node]
        roles_m.insert(0, node)
        all_data.append(roles_m)

    with open(out_file, "w") as f_out:
        for r in all_data:
            print(len(r))
            f_out.write(" ".join(map(str, r)) + "\n")
        print("saved node role_members.")
    # np.save(out_file + 'node_roleMembers.npy', node_roleMembers)  ###{}
    return

def plot_roles(roles_nodes):
    role_count = {}
    for i in roles_nodes:
        role_count[i] = len(roles_nodes[i])
    x = np.array(list(role_count.keys()) )
    y = np.array(list(role_count.values()))
    sns.barplot(x, y)
    plt.show()
    return

def read_word2vec_pairs(current_path, nviews):
    nodes_idx_nets = []
    for n_net in range(nviews):
        nodes_idx_nets.append(np.loadtxt(current_path + "/nodesidxPairs_" + str(n_net + 1) + ".txt", dtype=str))
    return nodes_idx_nets

def read_node_roleMembers(role_path):
    node_roleMembers = np.load(role_path, allow_pickle=True).item()
    return node_roleMembers

def save_role_pairs(pair_node, out_file):
    with open(out_file, 'w') as f:
        f.write(" ".join(map(str, pair_node)))
    return

def construct_role_pairs(path, nodes_idx_nets, structure_feature_list, graphs_roles, nview, view_id):
    nodes_idx = nodes_idx_nets#[view_id]
    print("nodes_idx length: ")
    print(len(nodes_idx))
    # view_ids = [i for i in range(len(nodes_idx_nets))]
    role_pairs_00 = []
    role_pairs_01 = []
    role_pairs_02 = []
    # if nview==3:
    for node in nodes_idx:
        node = str(node)
        roleOFnode = structure_feature_list[view_id][node][0]
        ####
        add_node_00 = add_one_node(graphs_roles,id=0, roleOFnode=roleOFnode,node=node)  ##
        role_pairs_00.append(add_node_00)

        add_node_01 = add_one_node(graphs_roles,id=1, roleOFnode=roleOFnode,node=node)  ##
        role_pairs_01.append(add_node_01)

        add_node_02 = add_one_node(graphs_roles,id=2, roleOFnode=roleOFnode,node=node)  ##
        role_pairs_02.append(add_node_02)

    print('pairs 0 length:  ')
    print(len(role_pairs_00))
    print('pairs 1 length:  ')
    print(len(role_pairs_01))
    print('pairs 2 length:  ')
    print(len(role_pairs_02))

    role_pairsidx_file_00 = path + "nodesidxPairs_" + str(view_id + 1) + '_' + '1' + ".txt"
    role_pairsidx_file_01 = path + "nodesidxPairs_" + str(view_id + 1) + '_' + '2' + ".txt"
    role_pairsidx_file_02 = path + "nodesidxPairs_" + str(view_id + 1) + '_' + '3' + ".txt"
    save_role_pairs(np.array(list(role_pairs_00)).squeeze(), role_pairsidx_file_00)
    save_role_pairs(np.array(list(role_pairs_01)).squeeze(), role_pairsidx_file_01)
    save_role_pairs(np.array(list(role_pairs_02)).squeeze(), role_pairsidx_file_02)

    return np.array(list(role_pairs_00)), np.array(list(role_pairs_01)), np.array(list(role_pairs_02))

def add_one_node(graphs_roles,id, roleOFnode,node):
    # print(roleOFnode)
    role_members = graphs_roles[id][roleOFnode]
    if len(role_members)>0:
        add_node = random.choices(role_members)[0]##
    else:
        add_node = node ##
    return int(add_node)

# if __name__ == '__main__':
#     args = parameter_parser()
#     input_graphs = r'./data\networks/'
#     dataset = 'Youtube/'   #  IntAct  LinkedIn  Youtube
#     out_file = r'./output\roles/'
#     pair_path = r'./data\pairs/'
#
#     G = read_graphs(input_graphs + dataset, 3)####nviews = 2,3
#     common_nodes = sorted(set(G[0]).intersection(*G))
#     print('Number of common/core nodes in all networks: ', len(common_nodes))
#     node2idx = {n: idx for (idx, n) in enumerate(common_nodes)}
#     idx2node = {idx: n for (idx, n) in enumerate(common_nodes)}
