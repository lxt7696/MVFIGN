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
from crypto_utils_2 import MPCBeaver
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
    t=[F.cosine_similarity(ba_ex1[i],ba_ex2[i], dim=1) for i in range(n)]
    return torch.stack(t)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def deal_dataset(client_num,rate,r):
    #dataset_str='Citeseer'
    #dataset_str='PubMed'
    dataset_str='PubMed'
    tdataset=Planetoid(root='torch_dataset/',name=dataset_str)
    tdata=tdataset[0]
    labels=tdata['y']
    data_num=len(tdata['x'])
    feature_num=len(tdata['x'][0])
    label_num=len(tdata['y'].unique())
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
    n_r=np.random.permutation(data_num)
    train_index=n_r[:int(0.7*data_num)].tolist()
    test_index=n_r[int(0.7*data_num):].tolist()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    edge_num=adj.shape[0]
    c_e_num=[i*edge_num for i in [0,0.9,1]] #边划分
    c_n_num=[i*feature_num for i in [0,1,0,1]]
    e_t=np.random.permutation(edge_num)
    #n_t=np.random.permutation(feature_num)
    n_t=np.array([i for i in range(feature_num)])
    c_edge_index=[]
    adj_list=[]
    for i in range(client_num):
        adj1=adj[train_index,:][:,train_index]
        c_edge_index.append(np.setdiff1d(e_t, e_t[int(c_e_num[i]):int(c_e_num[i+1])]))
        ee=np.array(adj1.todense()).nonzero()[0][c_edge_index[i]].tolist()
        ee1=np.array(adj1.todense()).nonzero()[1][c_edge_index[i]].tolist()
        for x,y in zip(ee,ee1):
            adj1[(x,y)]=0
        adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
        adj1=adj1.todense()
        adj_sp=sp.csr_matrix(adj1)
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj2 = adj_normalizer(adj_sp) 
        adj3 = sparse_mx_to_torch_sparse_tensor(adj2)
        adj4=get_A_r(adj3,1)
        adj_list.append((adj4).cuda())
#self
    user_features=[]
    user_labels=[]
    train_features=[]
    train_labels=[]
    for i in range(client_num):
        user_features.append([features[index].cuda() for index in train_index])
        # ↑ 每个client都得到完整的features，没有特征分割！
        train_features.append([features[index] for index in train_index])
    train_labels.append([labels[index] for index in train_index])
    test_features=[torch.stack([features[index][n_t[int(c_n_num[k]):int(c_n_num[k+1])]].cuda() for index in test_index]) for k in range(0,client_num+1,2)]
    test_features=torch.stack(test_features)
    test_labels=[labels[index] for index in test_index]
    at_features=[torch.stack([features[index][n_t[int(c_n_num[k]):int(c_n_num[k+1])]].cuda() for index in train_index]) for k in range(0,client_num+1,2)]
    at_labels=[labels[index] for index in train_index]
    at_features=torch.stack(at_features)
    # 新增: 初始化MPC工具
    mpc_beaver = MPCBeaver(client_num)
    return user_features,user_labels,adj_list,label_num,train_features,train_labels,test_features,test_labels,at_features,at_labels,mpc_beaver
