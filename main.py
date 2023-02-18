'''
The code is based on the work of Ata S K, Fang Y, Wu M, et al. Multi-view collaborative network embedding[J].
ACM Transactions on Knowledge Discovery from Data (TKDD), 2021, 15(3): 1-18. Their code: https://github.com/sezinata/MANE
'''
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import networkx as nx
import numpy as np
import generate_pairs
import random
import gc
import time
from sklearn import preprocessing
from args_parser import get_parser
from collections import OrderedDict
import generate_roles
import pickle
from utils import *

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run RMNE.")

    parser.add_argument("--min-count",
                        type=int,
                        default=1,
                        help="Minimal feature count. Default is 1.")

    parser.add_argument("--features",
                        nargs="?",
                        default="motif_tri",
                        help="Feature extraction mechanism. Default is motif_tri.")

    parser.add_argument("--labeling-iterations",
                        type=int,
                        default=2,
                        help="Number of WL labeling iterations. Default is 2.")

    parser.add_argument("--log-base",
                        type=int,
                        default=1.5,
                        help="Log base for label creation. Default is 1.5.")

    parser.add_argument("--graphlet-size",
                        type=int,
                        default=3,
                        help="Maximal graphlet size. Default is 4.")  ##

    parser.add_argument("--quantiles",
                        type=int,
                        default=5,
                        help="Number of quantiles for binning. Default is 5.")

    parser.add_argument("--motif-compression",
                        nargs="?",
                        default="string",
                        help="Motif compression procedure -- string or factorization.")

    parser.add_argument("--factors",
                        type=int,
                        default=8,
                        help="Number of factors for motif compression. Default is 8.")

    parser.add_argument("--clusters",
                        type=int,
                        default=50,
                        help="Number of motif based labels. Default is 50.")

    parser.add_argument("--beta",
                        type=float,
                        default=0.01,
                        help="Motif compression factorization regularizer. Default is 0.01.")

    parser.add_argument("--num_iters",
                        type=int,
                        default=2,
                        help="num_iters. Default is 2.")

    return parser.parse_args(args=[])

class RMNE(nn.Module):  # ready for cluster, no cache cleaning or loop shortening
    def __init__(self, params, len_common_nodes, embed_freq, batch_size, negative_sampling_size=10):
        super(RMNE, self).__init__()
        self.n_embedding = len_common_nodes
        self.embed_freq = embed_freq
        self.num_net = params.nviews
        self.negative_sampling_size = negative_sampling_size
        self.node_embeddings = nn.ModuleList()
        self.neigh_embeddings = nn.ModuleList()
        self.embedding_dim = params.dimensions
        self.device = params.device
        for n_net in range(self.num_net):  # len(G)
            self.node_embeddings.append(nn.Embedding(len_common_nodes, self.embedding_dim))
            self.neigh_embeddings.append(nn.Embedding(len_common_nodes, self.embedding_dim))

        self.batch_size = batch_size

    def forward(self, count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, node_role_nets, hyp1, hyp2, hyp3):

        # '''
        # Clear version of cost1 cost2 and cost3
        cost = []
        for i in range(self.num_net):

            batch_indices = shuffle_indices_nets[i][count:count + self.batch_size]
            nodes_idx = torch.LongTensor(nodes_idx_nets[i][batch_indices]).to(self.device)
            node_emb = self.node_embeddings[i](Variable(nodes_idx)).view(len(batch_indices), -1).unsqueeze(2)
            neighs_idx = torch.LongTensor(neigh_idx_nets[i][batch_indices]).to(self.device)
            neigh_emb = self.neigh_embeddings[i](Variable(neighs_idx)).unsqueeze(2).view(len(batch_indices), -1,
                                                                                         self.embedding_dim)
            loss_positive = nn.functional.logsigmoid(torch.bmm(neigh_emb, node_emb)).squeeze().mean()
            negative_context = self.embed_freq.multinomial(
                len(batch_indices) * neigh_emb.size(1) * self.negative_sampling_size,
                replacement=True).to(self.device)
            negative_context_emb = self.neigh_embeddings[i](negative_context).view(len(batch_indices), -1,
                                                                                   self.embedding_dim).neg()
            loss_negative = nn.functional.logsigmoid(torch.bmm(negative_context_emb, node_emb)).squeeze().sum(1).mean(0)
            cost.append(loss_positive + loss_negative)
            for j in range(self.num_net):
                if j != i:
                    node_neigh_emb = self.node_embeddings[j](Variable(nodes_idx)).unsqueeze(2).view(len(batch_indices),
                                                                                                    -1,
                                                                                                    self.embedding_dim)
                    loss_positive2 = nn.functional.logsigmoid(torch.bmm(node_neigh_emb, node_emb)).squeeze().mean()
                    negative_context2 = self.embed_freq.multinomial(
                        len(batch_indices) * node_neigh_emb.size(1) * self.negative_sampling_size,
                        replacement=True).to(self.device)
                    negative_context_emb2 = self.node_embeddings[j](negative_context2).view(len(batch_indices), -1,
                                                                                            self.embedding_dim).neg()
                    loss_negative2 = nn.functional.logsigmoid(torch.bmm(negative_context_emb2, node_emb)).squeeze().sum(
                        1).mean(0)
                    cost.append(hyp1 * (loss_positive2 + loss_negative2))
            for j in range(self.num_net):
                if j != i:
                    cross_neighs_idx = torch.LongTensor(
                        neigh_idx_nets[i][batch_indices]).to(self.device)
                    cross_neigh_emb = self.neigh_embeddings[j](Variable(cross_neighs_idx)).unsqueeze(2).view(
                        len(batch_indices), -1,
                        self.embedding_dim)
                    loss_positive3 = nn.functional.logsigmoid(torch.bmm(cross_neigh_emb, node_emb)).squeeze().mean()
                    negative_context3 = self.embed_freq.multinomial(
                        len(batch_indices) * cross_neigh_emb.size(1) * self.negative_sampling_size,
                        replacement=True).to(self.device)
                    negative_context_emb3 = self.neigh_embeddings[j](negative_context3).view(len(batch_indices), -1,
                                                                                             self.embedding_dim).neg()
                    loss_negative3 = nn.functional.logsigmoid(torch.bmm(negative_context_emb3, node_emb)).squeeze().sum(
                        1).mean(0)
                    cost.append(hyp2 * (loss_positive3 + loss_negative3))

            for j in range(self.num_net):
                ###
                role_neighs_idx = torch.LongTensor(node_role_nets[i][j][batch_indices]).to(self.device)  ##00, 10
                role_neigh_emb = self.node_embeddings[j](Variable(role_neighs_idx)).unsqueeze(2).view(
                    len(batch_indices), -1, self.embedding_dim)

                loss_positive4 = nn.functional.logsigmoid(torch.bmm(role_neigh_emb, node_emb)).squeeze().mean()
                negative_context4 = self.embed_freq.multinomial(
                    len(batch_indices) * role_neigh_emb.size(1) * self.negative_sampling_size, replacement=True).to(
                    self.device)
                negative_context_emb4 = self.node_embeddings[j](negative_context4).view(len(batch_indices), -1,
                                                                                        self.embedding_dim).neg()
                loss_negative4 = nn.functional.logsigmoid(torch.bmm(negative_context_emb4, node_emb)).squeeze().sum(
                    1).mean(0)

                cost.append(hyp3 * (loss_positive4 + loss_negative4))

        return -sum(cost) / len(cost)

def main():
    """
    Initialize parameters and train
    """
    params = get_parser().parse_args()
    print(params)
    args = parameter_parser()  ## generate roles

    if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, you may try cuda with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device
    print("Running on device: ", device)
    ####
    G = read_graphs(params.input_graphs + params.dataset, params.nviews)
    common_nodes = sorted(set(G[0]).intersection(*G))
    print('Number of common/core nodes in all networks: ', len(common_nodes))
    node2idx = {n: idx for (idx, n) in enumerate(common_nodes)}
    idx2node = {idx: n for (idx, n) in enumerate(common_nodes)}

    # relabeled_G = relabel_G(G, node2idx)

    if params.read_pair:
        nodes_idx_nets, neigh_idx_nets, node_role_nets = read_word2vec_pairs(params.input_pairs + params.dataset,params.nviews)
    else:
        nodes_idx_nets = []
        neigh_idx_nets = []
        node_role_nets = []
        #########################
        graphs_roles_pkl = params.output + params.dataset + 'graphs_roles.pkl'
        structure_feature_list_pkl = params.output + params.dataset + 'structure_feature_list.pkl'
        if os.path.exists(graphs_roles_pkl) and os.path.exists(structure_feature_list_pkl):
            print('read graphs_roles....')
            graphs_roles = read_data(graphs_roles_pkl)
            structure_feature_list = read_data(structure_feature_list_pkl)

            print(structure_feature_list[0]['0'])
        else:

            structure_feature_list, graphs_roles = generate_roles.structual_roles(args, G)

            print(structure_feature_list[0]['0'])
                                            
            save_data(graphs_roles_pkl,graphs_roles)
            save_data(structure_feature_list_pkl, structure_feature_list)

        view = []
        for n_net in range(params.nviews):
            view_id = n_net + 1
            print("View ", view_id)

            nodes_idx, neigh_idx = generate_pairs.construct_word2vec_pairs(G[n_net], view_id, common_nodes, params.p,
                                                                           params.q, params.window_size,
                                                                           params.num_walks,
                                                                           params.walk_length,
                                                                           params.output_pairs + params.dataset,
                                                                           node2idx)

            nodes_idx_nets.append(nodes_idx)
            neigh_idx_nets.append(neigh_idx)

            role_pairs_00, role_pairs_01, role_pairs_02 = generate_roles.construct_role_pairs(
                params.output_pairs + params.dataset, nodes_idx,  structure_feature_list, graphs_roles, 3, n_net)
            # view.append(role_pairs_00)
            # view.append(role_pairs_01)
            # view.append(role_pairs_02)

            node_role_nets.append([role_pairs_00, role_pairs_01, role_pairs_02])

    multinomial_nodes_idx = degree_nodes_common_nodes(G, common_nodes, node2idx)

    embed_freq = Variable(torch.Tensor(multinomial_nodes_idx))

    model = RMNE(params, len(common_nodes), embed_freq, params.batch_size)
    model.to(device)

    epo = 0
    min_pair_length = nodes_idx_nets[0].size
    for n_net in range(params.nviews):
        if min_pair_length > nodes_idx_nets[n_net].size:
            min_pair_length = nodes_idx_nets[n_net].size
    print("Total number of pairs: ", min_pair_length)
    print("Training started! \n")

    while epo <= params.epochs - 1:
        start_init = time.time()

        epo += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        running_loss = 0
        num_batches = 0
        shuffle_indices_nets = []
        fifty = False

        for n_net in range(params.nviews):
            shuffle_indices = [x for x in range(nodes_idx_nets[n_net].size)]
            random.shuffle(shuffle_indices)
            shuffle_indices_nets.append(shuffle_indices)
        for count in range(0, min_pair_length, params.batch_size):
            optimizer.zero_grad()
            loss = model(count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, node_role_nets, params.alpha,
                         params.beta, params.gamma)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            num_batches += 1
            if int(num_batches % 100) == 0:
                print(num_batches, " batches completed\n")
            elif not fifty and (count / min_pair_length) * 100 > 50:
                print("############# 50% epoch is completed #################\n")
                fifty = True
            torch.cuda.empty_cache()
            gc.collect()

        total_loss = running_loss / (num_batches)
        elapsed = time.time() - start_init
        print('epoch=', epo, '\t time=', elapsed, ' seconds\t total_loss=', total_loss)

    concat_tensors = model.node_embeddings[0].weight.detach().cpu()
    print('Embedding of view ', 1, ' ', concat_tensors)

    for i_tensor in range(1, model.num_net):
        print('Embedding of view ', (i_tensor + 1), ' ', model.node_embeddings[i_tensor].weight.detach().cpu())
        concat_tensors = torch.cat((concat_tensors, model.node_embeddings[i_tensor].weight.detach().cpu()), 1)

    emb_file = params.output + params.dataset + "Embedding_" + "concatenated_without_attention" + '_epoch_' + str(
        epo) + "_" + ".txt"
    embed_result = np.array(concat_tensors)
    fo = open(emb_file, 'a+')
    for idx in range(len(embed_result)):
        word = str((idx2node[idx]))
        fo.write(word + ' ' + ' '.join(
            map(str, embed_result[idx])) + '\n')
    fo.close()
    #
    with open(emb_file, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(str(len(common_nodes)) + ' ' + str(int(params.dimensions * params.nviews)) + '\n' + content)

if __name__ == '__main__':
    import time
    # dataset = 'higgs'
    #
    start = time.time()
    main()
    print('time：{}'.format(time.time()-start))

