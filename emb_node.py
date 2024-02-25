import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.data import Data

from embedder import embedder

from encoder import GCN, GCNLayer
from util.models.robcon import DGI
from util.utils import to_dense_subadj
from copy import deepcopy
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, subgraph, add_self_loops
import torch.nn.functional as F

from util.transforms import get_graph_drop_transform

class emb_node(embedder):
    def __init__(self, args,data1,data2,device):
        embedder.__init__(self, args,data1,data2,device)
        self.model = None
        self.args = args
        self.data1 = data1
        self.data2 = data2
        self.device = device
        #self.knn_data = knn_data
    def subgraph_sampling(self, data1, data2):
        self.sample_size = min(self.args.sub_size, self.args.n_node)
        nodes = torch.randperm(data1.x.size(0))[:self.sample_size].sort()[0]
        edge1, edge2 = add_self_loops(data1.edge_index, num_nodes=data1.x.size(0))[0], add_self_loops(data2.edge_index, num_nodes=data1.x.size(0))[0]
        edge1 = subgraph(subset=nodes, edge_index=edge1, relabel_nodes=True)[0]
        edge2 = subgraph(subset=nodes, edge_index=edge2, relabel_nodes=True)[0]
        tmp1, tmp2 = Data(), Data()
        tmp1.x, tmp2.x = data1.x[nodes], data2.x[nodes]
        tmp1.edge_index, tmp2.edge_index = edge1, edge2

        return tmp1, tmp2
    def subgraph_sampling_for_one(self, data1):

        self.sample_size = min(self.args.sub_size, self.args.n_node)
        nodes = torch.randperm(data1.x.shape[0])[:self.sample_size].sort()[0]
        edge1 = add_self_loops(data1.edge_index, num_nodes=data1.x.shape[0])[0]
        edge1 = subgraph(subset=nodes, edge_index=edge1, relabel_nodes=True)[0]
        tmp1= Data()
        tmp1.x = data1.x[nodes]
        tmp1.edge_index = edge1

        return tmp1
    def training(self,device,target_nodes=None,):
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)
        for seed in range(self.args.seed):
            print(f'is {seed} start')
            data = deepcopy(self.data1)
            data.edge_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0].to_sparse()#边索引转换为稀疏邻接矩阵
            data2 = deepcopy(self.data2)
            #knn_data = deepcopy(self.knn_data)
            #transform_1 = get_graph_drop_transform(drop_edge_p=self.args.d_1, drop_feat_p=self.args.d_1)
            if self.args.dataset == 'pumbed':
                transform_1 = get_graph_drop_transform(drop_edge_p=0.0, drop_feat_p=0.0)
                transform_2 = get_graph_drop_transform(drop_edge_p=0.0, drop_feat_p=0.0)
                transform_3 = get_graph_drop_transform(drop_edge_p=0.1, drop_feat_p=0.2)
            else:
                transform_1 = get_graph_drop_transform(drop_edge_p=0.0, drop_feat_p=0.2)
                transform_2 = get_graph_drop_transform(drop_edge_p=self.args.d_2, drop_feat_p=self.args.d_2)
                transform_3 = get_graph_drop_transform(drop_edge_p=0.1, drop_feat_p=0.2)
            transform_3 = get_graph_drop_transform(drop_edge_p=0.1, drop_feat_p=0.2)
            self.encoder = GCN(GCNLayer, [self.args.in_dim] + self.args.layers, batchnorm=self.args.bn).to(device) # 512, 128
            self.model = modeler(self.encoder, self.args.layers[-1], self.args.layers[-1], self.args.tau).to(device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            best, cnt_wait = 0, 0
            for epoch in range(1, self.args.epochs+1):
                sub1, sub2 = self.subgraph_sampling(data, data2)
                #sub3 = self.subgraph_sampling_for_one(knn_data)

                self.model.train()
                self.optimizer.zero_grad()
                #x1 = transform_1(sub2)
                x1 = transform_1(sub1)
                x2 = transform_2(sub2)
                #x3 = transform_3(sub3)
                #x4 = transform_3(sub1)
                x1 = x1.to(device)
                x2 = x2.to(device)
                #x3 = x3.to(device)
                #x4 = x4.to(device)
                x1.edge_adj, x2.edge_adj = to_dense_subadj(x1.edge_index, self.sample_size), to_dense_subadj(x2.edge_index, self.sample_size)
                #x3.edge_adj= to_dense_subadj(x1.edge_index, self.sample_size)
                #x4.edge_adj = to_dense_subadj(x4.edge_index, self.sample_size)
                x1.edge_adj = x1.edge_adj.requires_grad_()
                x2.edge_adj = x2.edge_adj.requires_grad_()
                #x3.edge_adj = x3.edge_adj.requires_grad_()
                #x4.edge_adj = x4.edge_adj.requires_grad_()
                x1.x = x1.x.requires_grad_()
                x2.x = x2.x.requires_grad_()
                #x3.x = x3.x.requires_grad_()
                #x4.x = x4.x.requires_grad_()
                z1 = self.model(x1.x, x1.edge_adj.to_sparse())
                z2 = self.model(x2.x, x2.edge_adj.to_sparse())
                #z3 = self.model(x3.x, x3.edge_adj.to_sparse())
                #z4 = self.model(x4.x, x4.edge_adj.to_sparse())
                loss = self.model.loss(z1, z2, batch_size=0)
                #loss += self.model.loss(z2, z3, batch_size=0) * self.args.lambda_1 *0.8
                #loss += self.model.loss(z2, z4, batch_size=0) * self.args.lambda_1 *0.5
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                print(f'Epoch {epoch}: Loss {loss.item()}')

                if epoch % self.args.verbose == 0:

                    val_acc = self.verbose(data)
                    if val_acc > best:
                        best = val_acc
                        cnt_wait = 0
                        torch.save(self.model.online_encoder.state_dict(), '{}/saved_model/best_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack_method, self.args.ptb_rate,seed))
                    else:
                        cnt_wait += self.args.verbose

                    if cnt_wait == self.args.patience:
                        print('Early stopping!')
                        break

            self.model.online_encoder.load_state_dict(torch.load('{}/saved_model/best_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack_method, self.args.ptb_rate, seed)))
            if self.args.save_embed:
                self.get_embeddings(data)
            if self.args.attack_type == 'poison':
                self.eval_poisoning(data)
            elif self.args.attack_type == 'evasive':
                self.eval_clean_and_evasive(data,True)

        self.summary_result()


        
class modeler(torch.nn.Module):
    def __init__(self, encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(modeler, self).__init__()
        self.online_encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        return self.online_encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,l2_lambda: float = 0.01):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        #l2_regularization = l2_lambda * (torch.sum(z1**2) + torch.sum(z2**2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))\
            #+l2_regularization

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int,l2_lambda: float = 0.01):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        #l2_regularization = l2_lambda * (torch.norm(z1, p=2) + torch.norm(z2, p=2))
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)#+l2_regularization

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
