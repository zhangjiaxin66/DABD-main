from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import torch
import random, os
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.data import Data

def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())



def get_reliable_neighbors(adj, features, k, degree_threshold):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features.to('cpu'))
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return

def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())
# def adj_new_norm(adj, alpha):
#     adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
#     degree = adj.sum(dim=1)
#     in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
#     out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
#     adj = sparse_dense_mul(adj, in_degree_norm)
#     adj = sparse_dense_mul(adj, out_degree_norm)
#     if alpha != -0.5:
#         return adj / (adj.sum(dim=1).reshape(adj.shape[0], -1))
#     else:
#         return adj


def preprocess_adj(features, adj,  metric='similarity', threshold=0.03, jaccard=True):
    """Drop dissimilar edges.(Faster version using numba)
    """

    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A  # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if jaccard:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                           threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                          threshold=threshold)
    print('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj


def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum()) + 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt

def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a * b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
            # if J > 0.4:
            #     A[i] = 1
            #     # A[n2, n1] = 0
            #     print("1")
            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)
            if C <= threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape))


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def idx_to_mask(idx, nodes_num):
    """Convert a indices array to a tensor mask matrix
    Args:
        idx : numpy.array
            indices of nodes set
        nodes_num: int
            number of nodes
    """
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.
    Args:
        tensor : torch.Tensor
                 given tensor
    Returns:
        bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:

        return True
    else:
        return False
def subgraph(subset,edge_index, edge_attr = None, relabel_nodes: bool = False):
    """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # if relabel_nodes:
    #     node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
    #                            device=device)
    #     node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    #     edge_index = node_idx[edge_index]


    return edge_index, edge_attr, edge_mask


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def to_dense_subadj(edge_index, subsize):
    edge = add_self_loops(edge_index, num_nodes=subsize)[0]
    return to_dense_adj(edge)[0].fill_diagonal_(0.0)

def get_reliable_neighbors(adj, features, k, degree_threshold):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features.to('cpu'))
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return
def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
def get_split(args,data, device,num_nodes):
    rs = np.random.RandomState(10)
    perm = rs.permutation(num_nodes)
    train_number = int(0.8*len(perm))
    #idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    idx_train = torch.tensor(sorted(perm[:train_number]), dtype=torch.long).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    #idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number]), dtype=torch.long).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True
    test_number = int(0.1 * len(perm))
    # idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    idx_test = torch.tensor(sorted(perm[train_number + val_number:train_number + val_number + test_number]),
                            dtype=torch.long).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True



    return data, idx_train, idx_val, idx_test

def to_dense_subadj(edge_index, subsize):
    edge = add_self_loops(edge_index, num_nodes=subsize)[0]
    return to_dense_adj(edge)[0].fill_diagonal_(0.0)
def subgraph_sampling(data1,n_node):
    sample_size = n_node
    nodes = torch.randperm(data1.x.size(0))[:sample_size].sort()[0]
    edge1 = add_self_loops(data1.edge_index, num_nodes=data1.x.size(0))[0]
    edge1 = subgraph(subset=nodes, edge_index=edge1, relabel_nodes=True)[0]

    tmp1 = Data()
    tmp1.x = data1.x[nodes]
    tmp1.edge_index = edge1

    return tmp1
def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index,
                                    torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])


# def idx_to_mask(idx, nodes_num):
#     """Convert a indices array to a tensor mask matrix
#     Args:
#         idx : numpy.array
#             indices of nodes set
#         nodes_num: int
#             number of nodes
#     """
#     mask = torch.zeros(nodes_num)
#     mask[idx] = 1
#     return mask.bool()
# def to_scipy(tensor):
#     # """Convert a dense/sparse tensor to scipy matrix"""
#     if torch.is_sparse(tensor):
#         values = tensor.values().cpu().numpy()
#         indices = tensor.indices().t().cpu().numpy()
#         return sp.csr_matrix((values, indices), shape=tensor.shape)
#     else:
#         indices = tensor.nonzero().t()
#         values = tensor[indices[0], indices[1]]
#         return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)




def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def dense_to_sparse_adj(edge_index, n_node):
    return torch.sparse.FloatTensor(edge_index,
                                    torch.ones(edge_index.shape[1]).to(edge_index.device),
                                    [n_node, n_node])


def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index,
                                    torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device', 'patience', 'epochs', 'save_dir', 'in_dim', 'n_class', 'best_epoch', 'save_fig',
                        'n_node', 'n_degree', 'attack', 'attack_type', 'ptb_rate', 'verbose', 'mm', '']:
            st_ = "{}:{} / ".format(name, val)
            st += st_

    return st[:-1]

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    #sparse_data = torch.stack(list(sparse_mx.data))


    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape))

import torch

class Standard:
    def __init__(self,preds,labels):
        self.nclass = labels.max().item()+1
        self.labelClassNum = []
        self.correctClassNum = []
        self.predsClassNum = []
        for i in range(self.nclass):
            labelClassIndex = (labels == i).sum().item()
            predsClassIndex = (preds == i).sum().item()
            self.labelClassNum.append(labelClassIndex if labelClassIndex != 0 else 1)
            self.predsClassNum.append(predsClassIndex if predsClassIndex != 0 else 1)
            self.correctClassNum.append(((labels == i) & (preds == i)).sum().item())
        self.labelClassNum = torch.tensor(self.labelClassNum)
        self.correctClassNum = torch.tensor(self.correctClassNum)
        self.predsClassNum = torch.tensor(self.predsClassNum)

    def precision(self):
        return (self.correctClassNum / self.predsClassNum).mean().item()
    def recall(self):
        return (self.correctClassNum / self.labelClassNum).mean().item()
    def f1(self):
        return 2 / (1/self.precision() + 1 / self.recall())