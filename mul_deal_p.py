import sys
import os
import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
def tensor_intersect(t1, t2):
    indices = torch.zeros_like(t1, dtype = torch.bool)
    for elem in t2:
        indices = indices | (t1 == elem)  
        intersection = t1[indices]  
    return intersection
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func
def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label
def get_cos(features,index):
    i_features=torch.stack(features)[index]
    n=len(index)
    ba_ex=torch.unsqueeze(i_features.cpu(),1)
    ba_ex1 =ba_ex.repeat(1, n, 1)
    ba_ex2=i_features.repeat(n, 1, 1)
    print(ba_ex2)
    t=[F.cosine_similarity(ba_ex1[i],ba_ex2[i], dim=1) for i in range(n)]
    print(t)
    return torch.stack(t)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def deal_dataset(client_num,rate,r):
    tdataset = PygNodePropPredDataset(name='ogbn-products', root='./products/')
    tdata=tdataset[0]
    data_num=len(tdata.x)
    feature_num=len(tdata.x[0])
    label_num=len(tdata.y.unique())
    split_idx = tdataset.get_idx_split()
    train_index = split_idx['train'].tolist()
    val_index = split_idx['valid'].tolist()
    test_index = split_idx['test'].tolist()
    edge_num=tdata.edge_index.shape[1]
    features=[torch.from_numpy(i) for i in row_normalize(tdata.x)]
    user_data_index=[]
    adj=coo_matrix((np.ones(edge_num),(tdata.edge_index)),shape=(data_num,data_num), dtype=np.float32)
    adj_list_list=[]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    for j in range(r):
        adj_list=[]
        for i in range(client_num):
            adj1=adj.todense()
            adj_sp=sp.csr_matrix(adj1)
            adj_normalizer = fetch_normalization("AugNormAdj")
            adj2 = adj_normalizer(adj_sp) 
            adj3 = sparse_mx_to_torch_sparse_tensor(adj2)
            adj4=get_A_r(adj3,j+1)
            adj_list.append((adj4))
        adj_list_list.append(adj_list)
    print(adj_list_list)
#self
    user_features=[]
    user_labels=[]
    user_features=[]
    user_labels=[]
    train_features=[]
    train_labels=[]
    for i in range(client_num):
        user_features.append([i.cuda() for i in features])
        user_labels.append([i for i in tdata.y])
        
    test_features=[features[index].cuda() for index in test_index]
    val_features=[features[index].cuda() for index in val_index]
    test_features=torch.stack(test_features)
    test_labels=[tdata.y[index] for index in test_index]
    at_features=[features[index].cuda() for index in train_index]
    at_labels=[tdata.y[index] for index in train_index]
    at_features=torch.stack(at_features)
    val_features=torch.stack(val_features)
    val_labels=[tdata.y[index] for index in val_index]
    return user_features,user_labels,adj_list_list,label_num,train_features,train_labels,test_features,test_labels,val_features,val_labels,at_features,at_labels
