from copy import deepcopy

import numpy as np

from util.data import get_data
from util.utils import config2string, ensure_dir, to_numpy, dense_to_sparse_adj, Standard
import os

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
class embedder:
    def __init__(self, args,data1,data2,device):

        self.sprf = {}
        print('===',args.dataset, '===')
        self.data1 = data1
        self.data2 = data2
        #self.knn_data = knn_data
        self.args = args
        self.device = device
        # save results
        self.config_str = config2string(args)
        self.result_path = f'{args.save_dir}/summary_results/{args.dataset}.txt'
        # basic statistics
        self.args.in_dim = self.data1.x.shape[1]
        self.args.n_class = self.data1.y.unique().size(0)
        self.args.n_node = self.data1.x.shape[0]
        self.embed_dim = args.layers[-1]

        # save path check
        ensure_dir(f'{args.save_dir}/fig/{args.dataset}/')
        ensure_dir(f'{args.save_dir}/saved_model/')
        ensure_dir(f'{args.save_dir}/embeddings/')
        ensure_dir(f'{args.save_dir}/summary_result/bypass/')

    def get_embeddings(self, data):
        tmp_encoder = deepcopy(self.model.online_encoder).eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        torch.save(embed.cpu(), f'{self.args.save_dir}/embeddings/{self.args.dataset}_{self.args.attack_type}_{self.args.attack_method}_{self.args.ptb_rate}_embed_seed{self.args.seed}.pt')

    def eval_base(self, data):
        online_encoder = deepcopy(self.model.online_encoder)
        tmp_encoder = online_encoder.eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        embed = F.normalize(embed, dim=1, p=2)
        embed = to_numpy(embed)
        y = to_numpy(data.y)
        
        if len(data.train_mask.size()) == 2:
            train_mask, val_mask, test_mask = to_numpy(data.train_mask[self.seed, :]), to_numpy(data.val_mask[self.seed, :]), to_numpy(data.test_mask[self.seed, :])
        else:
            train_mask, val_mask, test_mask = to_numpy(data.train_mask), to_numpy(data.val_mask), to_numpy(data.test_mask)
        
        linear_model = LogisticRegression(solver='liblinear', multi_class='auto', class_weight=None)
        linear_model.fit(embed[train_mask], y[train_mask])
        pred = linear_model.predict(embed)
        return pred, y, train_mask, val_mask, test_mask
    
    def verbose(self, data):

        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        train_acc = np.mean(correct[train_mask])
        val_acc = np.mean(correct[val_mask])
        test_acc = np.mean(correct[test_mask])

        print(f'====== Train acc {train_acc*100:.2f}, Val acc {val_acc*100:.2f}, Test acc {test_acc*100:.2f},')
        return val_acc
    
    def verbose_link(self, data):

        tmp_encoder = deepcopy(self.model.online_encoder).eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        embed = F.normalize(embed, dim=1, p=2)
        embed = to_numpy(embed)
        
        src, dst = data.train_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.train_label.cpu().numpy()
        linear_model = LogisticRegression(solver='lbfgs')
        linear_model.fit(link_emb, y)
        pred = linear_model.predict_proba(link_emb)[:, 1]
        train_auc = roc_auc_score(y, pred)
        
        src, dst = data.val_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.val_label.cpu().numpy()
        pred = linear_model.predict_proba(link_emb)[:, 1]
        val_auc = roc_auc_score(y, pred)
        
        src, dst = data.test_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.test_label.cpu().numpy()
        pred = linear_model.predict_proba(link_emb)[:, 1]
        test_auc = roc_auc_score(y, pred)
        
        print(f'====== Train AUC {train_auc*100:.2f}, Val AUC {val_auc*100:.2f}, Test AUC {test_auc*100:.2f},')

        return train_auc, val_auc, test_auc
    
    def eval_link(self, data):
        save_dict = {'config':self.config_str}
        
        train_auc, val_auc, test_auc = self.verbose_link(data)
        self.train_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(train_auc)
        self.val_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(val_auc)
        self.test_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(test_auc)
    
    def eval_clean_and_evasive(self,data,only_clean=False):
        print("===========clean===========")
        save_dict = {'config': self.config_str}
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        s = Standard(pred, y)
        precision = s.precision()
        recall = s.recall()
        f1 = s.f1()
        print("precision:{:.4f},recall:{:.4f},f1:{:.4f}".format(precision, recall, f1))
        self.sprf = [[], [], []]
        self.sprf[0].append(np.mean(f1))
        self.sprf[1].append(np.mean(precision))
        self.sprf[2].append(np.mean(recall))
        ptb=0
        self.train_result['CLEAN'].append(np.mean(correct[train_mask]))
        self.val_result['CLEAN'].append(np.mean(correct[val_mask]))
        self.test_result['CLEAN'].append(np.mean(correct[test_mask]))
        save_dict['clean'] = [pred, y, train_mask, val_mask, test_mask]
        print(
            f'Train Acc: {np.mean(correct[train_mask]) * 100:.2f}, Val Acc: {np.mean(correct[val_mask]) * 100:.2f}, Test Acc: {np.mean(correct[test_mask]) * 100:.2f}')
        if not only_clean:
            if self.args.attack_method == 'meta': iterator = [0.05, 0.1, 0.15, 0.2, 0.25]
            if self.args.attack_method == 'nettack': iterator = [1.0, 2.0, 3.0, 4.0, 5.0]
            if self.args.attack_method == 'random': iterator = [0.2, 0.4, 0.6, 0.8, 1.0]
            for ptb in iterator:
                print("====evasive====")
                data_home = f'./data'
                data = get_data(data_home, self.args.dataset, self.args.attack_method, ptb,self.args.seed)[0]
                data = data.cuda()
                data.edge_adj = dense_to_sparse_adj(data.edge_index, data.x.size(0))
                pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
                correct = (pred == y)
                print(f'Train Acc: {np.mean(correct[train_mask])*100:.2f}, Val Acc: {np.mean(correct[val_mask])*100:.2f}, Test Acc: {np.mean(correct[test_mask])*100:.2f}')
                self.train_result[f'EVASIVE_{self.args.attack_method}_{ptb}'].append(np.mean(correct[train_mask]))
                self.val_result[f'EVASIVE_{self.args.attack_method}_{ptb}'].append(np.mean(correct[val_mask]))
                self.test_result[f'EVASIVE_{self.args.attack_method}_{ptb}'].append(np.mean(correct[test_mask]))
                save_dict[f'evasive_{self.args.attack_method}_{ptb}'] = [pred, y, train_mask, val_mask, test_mask]

        torch.save(save_dict, f'{self.args.save_dir}/summary_result/bypass/{self.args.dataset}_{self.args.attack_method}_{ptb}_save_dict_evasive_seed{self.args.seed}.pt')

    def eval_poisoning(self, data):

            save_dict = {'config':self.config_str}
            pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
            correct = (pred==y)
            ##自己加#
            s = Standard(pred, y)
            precision = s.precision()
            recall = s.recall()
            f1 = s.f1()
            print("precision:{:.4f},recall:{:.4f},f1:{:.4f}".format(precision, recall, f1))
            self.sprf = [[], [], []]
            # self.sprf[f'POISON_f1{self.args.attack_method}_{self.args.ptb_rate}'].append(np.mean(f1))
            # self.sprf[f'POISON_precision{self.args.attack_method}_{self.args.ptb_rate}'].append(precision)
            # self.sprf[f'POISON_recall{self.args.attack_method}_{self.args.ptb_rate}'].append(recall)
            self.sprf[0].append(np.mean(f1))
            self.sprf[1].append(np.mean(precision))
            self.sprf[2].append(np.mean(recall))

            ##自己加end#
            train_acc = np.mean(correct[train_mask])
            val_acc = np.mean(correct[val_mask])
            test_acc = np.mean(correct[test_mask])
            #all_acc = sum(correct)/correct.size(0)
            print("====eval poison====")
            print('Train Acc: {train_acc*100:.2f}, Val Acc: {val_mean*100:.2f}, Test Acc: {test_mean*100:.2f}',train_acc,val_acc,test_acc)
            print("到多少扰动率了")
            print(str(self.args.ptb_rate))
            self.train_result[f'POISON_{self.args.attack_method}_{self.args.ptb_rate}'].append(np.mean(train_acc))
            self.val_result[f'POISON_{self.args.attack_method}_{self.args.ptb_rate}'].append(np.mean(val_acc))
            self.test_result[f'POISON_{self.args.attack_method}_{self.args.ptb_rate}'].append(np.mean(test_acc))
            #self.all_result[f'POISON_{self.args.attack_method}_{self.args.ptb_rate}'].append(np.mean(all_acc))
            save_dict[f'poison_{self.args.attack_method}_{self.args.ptb_rate}'] = [pred, y, train_mask, val_mask, test_mask]
            torch.save(save_dict,
                       f'{self.args.save_dir}/summary_result/bypass/{self.args.dataset}_{self.args.attack_method}_{self.args.ptb_rate}_save_dict_poison_seed{self.args.seed}.pt')

    def summary_result(self):
        
        assert self.train_result.keys() == self.val_result.keys() and self.train_result.keys() == self.test_result.keys()
        key = list(self.train_result.keys())

        result_path = f'{self.args.save_dir}/summary_result/{self.args.dataset}_{self.args.attack_method}_{self.args.attack_type}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(self.config_str)
            f.write(f'\n')
            for k in key:
                f.write(f'====={k}=====')
                f.write(f'\n')
                train_mean, train_std = np.mean(self.train_result[k]), np.std(self.train_result[k])
                val_mean, val_std = np.mean(self.val_result[k]), np.std(self.val_result[k])
                test_mean, test_std = np.mean(self.test_result[k]), np.std(self.test_result[k])
                f.write(f'Train Acc: {train_mean*100:.2f}±{train_std*100:.2f}, Val Acc: {val_mean*100:.2f}±{val_std*100:.2f}, Test Acc: {test_mean*100:.2f}±{test_std*100:.2f}')
                f.write(f'\n')
                if self.args.attack_method == 'poison' and self.args.ptb_rate == 0.25:
                    f.write(f'='*40)
                    f.write(f'\n')
                else:
                    f.write(f'-'*2)
                    f.write(f'\n')
        result2_path = f'{self.args.save_dir}/summary_result/{self.args.dataset}_{self.args.attack_method}_{self.args.attack_type}_spf.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        #key2 = list(self.sprf.keys())
        with open(result2_path, mode) as f:
            f.write(self.config_str)
            f.write(f'\n')

            f.write(f'====={k}=====')
            f.write(f'\n')
            precision_mean, precision_std = np.mean(self.sprf[0]), np.std(self.sprf[0])
            recall_mean, recall_std = np.mean(self.sprf[1]), np.std(self.sprf[1])
            f1_mean, f1_std = np.mean(self.sprf[2]), np.std(self.sprf[2])
            f.write(
                    f'precision: {precision_mean * 100:.2f}±{precision_std * 100:.2f}, recall: {recall_mean * 100:.2f}±{recall_std * 100:.2f}, f1: {f1_mean * 100:.2f}±{f1_std * 100:.2f}')
            f.write(f'\n')
            if self.args.attack_method == 'poison' and self.args.ptb_rate == 0.25:
                    f.write(f'=' * 40)
                    f.write(f'\n')
            else:
                    f.write(f'-' * 2)
                    f.write(f'\n')

