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
    elif r == 5:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label
    elif r == 6:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label@adj_label
    elif r == 7:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label
    elif r == 8:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label
    elif r == 9:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label
    elif r == 10:
        adj_label = adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label@adj_label
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
    #dataset_str='Citeseer'
    dataset_str='PubMed'
    #dataset_str='Cora'
    tdataset=Planetoid(root='torch_dataset/',name=dataset_str)
    tdata=tdataset[0]
    data_num=len(tdata['x'])
    feature_num=len(tdata['x'][0])
    label_num=len(tdata['y'].unique())
    train_index=[]
    test_index=[]
    val_index=[]
    # 2. 使用数据集自带的mask划分
    for i in range(data_num):
        if tdata['train_mask'][i]==True:
            train_index.append(i)
        if tdata['test_mask'][i]==True:
            test_index.append(i)
        if tdata['val_mask'][i]==True:
            val_index.append(i)

    features=[torch.from_numpy(i) for i in row_normalize(tdata['x'])]
    user_data_index=[]
    with open("/home/lxt/project/MVFIGN/torch_dataset/PubMed/raw/ind.pubmed.graph", 'rb') as f:
    #with open("/home/lxt/project/MVFIGN/torch_dataset/Citeseer/raw/ind.citeseer.graph", 'rb') as f:
    #with open("/home/lxt/project/MVFIGN/torch_dataset/Cora/raw/ind.cora.graph", 'rb') as f:
            if sys.version_info > (3, 0):
                graph=pkl.load(f, encoding='latin1')
            else:
                graph=pkl.load(f)

#######################################
    if dataset_str == 'Citeseer':
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.tx", 'rb') as f1:
            if sys.version_info > (3, 0):
                tx=pkl.load(f1, encoding='latin1')
            else:
                tx=pkl.load(f1)
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.ty", 'rb') as f2:
            if sys.version_info > (3, 0):
                ty=pkl.load(f2, encoding='latin1')
            else:
                ty=pkl.load(f2)
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.x", 'rb') as f1:
            if sys.version_info > (3, 0):
                x=pkl.load(f1, encoding='latin1')
            else:
                x=pkl.load(f1)
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.y", 'rb') as f2:
            if sys.version_info > (3, 0):
                y=pkl.load(f2, encoding='latin1')
            else:
                y=pkl.load(f2)
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.allx", 'rb') as f1:
            if sys.version_info > (3, 0):
                allx=pkl.load(f1, encoding='latin1')
            else:
                allx=pkl.load(f1)
        with open("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.ally", 'rb') as f2:
            if sys.version_info > (3, 0):
                ally=pkl.load(f2, encoding='latin1')
            else:
                ally=pkl.load(f2)
        test_idx_reorder = parse_index_file("/home/lxt/project/MVFIGN/torch_dataset/CiteSeer/raw/ind.citeseer.test.index")
        test_idx_range = np.sort(test_idx_reorder)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        t=sp.vstack((allx, tx)).tolil()
        t[test_idx_reorder, :] = t[test_idx_range, :]
        features=torch.from_numpy(row_normalize(t).todense().astype(np.float32)) 
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = torch.LongTensor(labels)
        labels = torch.max(labels, dim=1)[1]
#######################################
    # 3. 邻接矩阵包含全图
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
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
            # 注意：j+2 表示从 2-hop 开始
            # j=0 -> 2-hop, j=1 -> 3-hop, j=2 -> 4-hop, j=3 -> 5-hop
            adj4=get_A_r(adj3,j+2)
            adj_list.append((adj4))
        adj_list_list.append(adj_list)
#self
    user_features=[]
    user_labels=[]
    user_features=[]
    user_labels=[]
    train_features=[]
    train_labels=[]
    for i in range(client_num):
        #1. 加载所有节点的features
        user_features.append([i.cuda() for i in features])
        user_labels.append([i for i in tdata['y']])
        
    test_features=[features[index].cuda() for index in test_index]
    val_features=[features[index].cuda() for index in val_index]
    test_features=torch.stack(test_features)
    test_labels=[tdata['y'][index] for index in test_index]
    at_features=[features[index].cuda() for index in train_index]
    at_labels=[tdata['y'][index] for index in train_index]
    at_features=torch.stack(at_features)
    val_features=torch.stack(val_features)
    val_labels=[tdata['y'][index] for index in val_index]
    return user_features,user_labels,adj_list_list,label_num,train_features,train_labels,test_features,test_labels,val_features,val_labels,at_features,at_labels
