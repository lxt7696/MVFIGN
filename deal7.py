import sys
import os
import random
import numpy as np
from collections import defaultdict
import torch

import scipy.sparse as sp
import pickle as pkl
import networkx as nx
from torch_geometric.datasets import Planetoid
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
    print(mx)
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
def deal_dataset(client_num,rate):
    tdataset=Planetoid(root='torch_dataset/',name='PubMed')
    tdata=tdataset[0]
    data_num=len(tdata.x)
    feature_num=len(tdata.x[0])
    label_num=len(tdata.y.unique())
    train_index=[]
    test_index=[]
    val_index=[]
    for i in range(data_num):
        if tdata.train_mask[i]==True:
            train_index.append(i)
        if tdata.test_mask[i]==True:
            test_index.append(i)
        if tdata.val_mask[i]==True:
            val_index.append(i)

    features=[torch.from_numpy(i) for i in row_normalize(tdata.x)]
    user_data_index=[]
    with open("/home/yaqi/FL_gnn/torch_dataset/PubMed/raw/ind.pubmed.graph", 'rb') as f:
            if sys.version_info > (3, 0):
                graph=pkl.load(f, encoding='latin1')
            else:
                graph=pkl.load(f)
    adj_list=[]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    user_data_index.append(torch.nonzero(tdata.y==1).squeeze())
    user_data_index.append(torch.nonzero(tdata.y!=1).squeeze())
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    for i in range(client_num):
        adj1=adj.todense()[user_data_index[i],:][:,user_data_index[i]]
        adj_sp=sp.csr_matrix(adj1)
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj2 = adj_normalizer(adj_sp) 
        adj3 = sparse_mx_to_torch_sparse_tensor(adj2)
        print(adj3.to_dense(),sum(adj3.to_dense()[0]))
        adj4=get_A_r(adj3,2)
        adj_list.append(adj4.cuda())
#self
    user_features=[]
    user_labels=[]
    user_train_index=[]

    #t_num=[i*data_num for i in [0,1]]
    #t_num=[i*data_num for i in [rate]]
    #t=np.random.choice(data_num, data_num,replace=False)
    #for i in range(client_num):
      #  t=np.random.choice(data_num,int(t_num[i]),replace=False)
        #user_data_index.append(t)
    #t=np.random.permutation(data_num)
    #for i in range(client_num):
      #  user_data_index.append(t[int(t_num[i]):int(t_num[i+1])])
    #for i in range(client_num):
     #   user_data_index.append(np.array([i for i in range(data_num)]))
    for i in range(client_num):
        user_data_list=user_data_index[i].numpy().tolist()
        print(i,len(user_data_list))
        user_train_index.append([user_data_list.index(j) for j in train_index if j in user_data_list])
    user_features=[]
    user_labels=[]
    train_features=[]
    train_labels=[]
    for i in range(client_num):
        user_features.append([features[index].cuda() for index in user_data_index[i]])
        user_labels.append([tdata.y[index] for index in user_data_index[i]])
        train_features.append([user_features[i][index] for index in user_train_index[i]])
        train_labels.append([user_labels[i][index] for index in user_train_index[i]])
        
    test_features=[features[index].cuda() for index in test_index]
    val_features=[features[index].cuda() for index in val_index]
    test_features=torch.stack(test_features)
    test_labels=[tdata.y[index] for index in test_index]
    at_features=[features[index].cuda() for index in train_index]
    at_labels=[tdata.y[index] for index in train_index]
    at_features=torch.stack(at_features)
    val_features=torch.stack(val_features)
    val_labels=[tdata.y[index] for index in val_index]

    #client_adj=[]
    #for i in range(client_num):
      #  client_adj.append(adj[user_data_index[i],:][:,user_data_index[i]].cuda())
    return user_features,adj_list,label_num,train_features,train_labels,test_features,test_labels,val_features,val_labels,at_features,at_labels
