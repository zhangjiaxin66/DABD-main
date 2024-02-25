import argparse
import warnings
from copy import deepcopy


import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse
import torch.nn.functional as F
from util.data import get_data
from emb_node import emb_node
from util.utils import preprocess_adj
import heuristic_selection as hs
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--jt', type=float, default=0.03,  help='jaccard threshold')
parser.add_argument('--attack_type', type=str, default='poison')#evasive  poison
parser.add_argument('--attack_method', type=str, default='meta')#nettack #meta #random
parser.add_argument('--dis_weight', type=float, default=0.3,
                    help="Weight of cluster distance")
parser.add_argument('--vs_number_rate', type=float, default=0.2,
                    help="number of select node rate")
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=150, help='Number of epochs to train model.')
parser.add_argument('--d_1', type=float, default=0.2)
parser.add_argument('--d_2', type=float, default=0.3)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--save_fig', action='store_true', default=True)
parser.add_argument("--save_embed", action='store_true', default=True)
parser.add_argument("--layers", nargs='*', type=int, default=[512, 128],
                    help="The number of units of each layer of the GNN. Default is [256].For emb_node.")
parser.add_argument("--bn", action='store_false', default=True)
parser.add_argument('--sub_size', type=int, default=5000)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--verbose', type=int, default=10)
parser.add_argument('--patience', type=int, default=400)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate. For kmean')
#parser.add_argument('--lambda_1', type=float, default=2.0)
parser.add_argument('--knn', type=int, default=10)
args = parser.parse_args()

# Loading data
data_home = f'./data'
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
# Training parameters
epochs = args.epochs
n_hidden = args.hidden
dropout = args.dropout
weight_decay = args.weight_decay
lr = args.lr
loss = nn.CrossEntropyLoss()

if args.dataset == 'pubmed':
    args.jt = 0.1
    args.vs_number_rate=0.4
    args.d_1 = 0.4
    args.d_2 = 0.3

if args.attack_method == 'nettack':
    args.ptb_rate = args.ptb_rate * 20
if args.attack_type == 'evasive':
    args.ptb_rate = 0.0

data = get_data(data_home, args.dataset, args.attack_method, args.ptb_rate,args.seed)[0]

n_node = data.y.shape[0]
n_class = data.y.max() + 1
if __name__ == '__main__':
    print(args)
    print('===start preprocessing the graph===')
    features = data.x
    perturbed_adj = data.adj
    adj_pre = preprocess_adj(features, perturbed_adj, threshold=args.jt)
    print('===end preprocessing the graph===')
    size = (int)(args.vs_number_rate * n_node)
    #暂时没有加入修改后的图，直接是原图得到得
    data.edge_index = dense_to_sparse(torch.from_numpy(adj_pre.toarray()))[0].long()
    #print(size)
    #print("#Attach Nodes:{}".format(size))
    k =args.knn
    features_tensor = torch.FloatTensor(data.x)
    sim = F.normalize(features_tensor).mm(F.normalize(features_tensor).T).fill_diagonal_(0.0)
    dst = sim.topk(k, 1)[1]  # 通过调用topk方法，选择相似度矩阵中每行的前self.args.knn个最大值，并存储它们的索引。
    src = torch.arange(features_tensor.size(0)).unsqueeze(1).expand_as(sim.topk(k, 1)[1])
    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)])
    edge_index = to_undirected(edge_index)
    #knn_edge_index = torch.cat([edge_index, data.edge_index], dim=1)
    knn_edge_index = to_undirected(edge_index)
    knn_data = Data(features_tensor,knn_edge_index)
    knn_data = knn_data.cuda()
    data.edge_index = torch.cat([knn_data.edge_index.to('cpu'), data.edge_index], dim=1)
    data.num_edges = knn_data.num_edges + data.num_edges

    data_new = deepcopy(data)
    data_new.edge_index = dense_to_sparse(torch.from_numpy(adj_pre.toarray()))[0].long()
    data_new.edge_index = to_undirected(data_new.edge_index)
    idx_train = np.where(data.train_mask)[0]
    idx_val = np.where(data.val_mask)[0]
    idx_test = np.where(data.test_mask)[0]
    idx_attach = hs.cluster_degree_selection(args, data_new, idx_train, idx_val, idx_test,data_new.edge_index, size, device,n_node)
    idx_attach = torch.LongTensor(idx_attach).to(device)
    #print("idx_attach: {}".format(idx_attach))
    # drop
    for idx in idx_attach:
        mask = (data_new.edge_index[0] != idx.to('cpu')) & (data_new.edge_index[1] != idx.to('cpu'))
        new_edge_index = data_new.edge_index[:, mask]
        data_new.edge_index = new_edge_index
        data_new.num_edges = data_new.edge_index.size(1)
        #print(data_new.num_edges)
    #print(data_new.edge_index)
    print('===start getting contrastive embeddings===')
    embedder = emb_node(args,data_new, data, device)
    embedder.training(device)

