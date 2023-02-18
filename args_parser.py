import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_pair',nargs='?',  default=True,
                        help='Default is true. If true, enables you to use the pairs you already have with your own deepwalk/node2vec setting')

    parser.add_argument('--input_graphs', nargs='?', default=r'./data/networks/',
                        help='Input graph path ')

    parser.add_argument('--input_pairs', nargs='?', default=r'./data/pairs/',
                        help='Input pairs path')

    parser.add_argument('--output', nargs='?', default=r'./output/emb/',
                        help='Embeddings path')

    parser.add_argument('--output_pairs', nargs='?', default=r'./output/pairs/',
                        help='Pairs output path')

    parser.add_argument('--dataset', nargs='?', default='LinkedIn/', #
                        help='Input graph path ')

    parser.add_argument('--nviews', type=int, default=3,
                        help='Number of views in dataset, i.e, if there are two networks nviews=3. Default is 3.')

    parser.add_argument('--dimensions', type=int, default=42,
                        help='Number of dimensions. Default is 128/|V|, i.e., 42 for 3 views, 128 for 1 view/network.')

    parser.add_argument('--alpha', type=float, default=1,
                        help='Hyperparameter for 1st order collaboration. Default is 1.')

    parser.add_argument('--beta', type=float, default=1,
                        help='Hyperparameter for 2nd order collaboration. Default is 1.')

    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Hyperparameter for 3nd order collaboration. Default is 1.')

    parser.add_argument('--walk_length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=5,
                        help='Number of walks per node. Default is 5.')

    parser.add_argument('--window_size', type=int, default=3,
                        help='Context size for optimization. Default is 3.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.add_argument('-lr', '--learning_rate',type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-ns', '--negative_sampling',type=float,
                        help='learning rate for the model, default=10',
                        default=10)

    parser.add_argument('-bs', '--batch_size',type=float,
                        help='batch size for the model, default=256',
                        default=256)

    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of training epochs',
                        default=10)
    parser.add_argument('--cuda',
                        action='store_false',
                        help='enables cuda. Default cuda.')
    return parser