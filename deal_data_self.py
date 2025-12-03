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

def deal_dataset(path,dataset,client_num):
    tdataset=Planetoid(root='torch_dataset/',name='Cora')
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
    with open("/home/lxt/project/MVFIGN/torch_dataset/Cora/raw/ind.cora.graph", 'rb') as f:
            if sys.version_info > (3, 0):
                graph=pkl.load(f, encoding='latin1')
            else:
                graph=pkl.load(f)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_normalizer = fetch_normalization("AugNormAdj")
    adj = adj_normalizer(adj) 
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

#self
    user_data_index=[]
    user_features=[]
    user_labels=[]
    user_train_index=[]
#random sample [0.7,0.6,...]
    #t_num=[i*data_num for i in [0.7,0.6,0.5,0.5,0.4,0.3]]
    t_num=[i*data_num for i in [1]]
    user_data_index.append([i for i in range(data_num)])
    #for i in range(client_num):
      #  user_data_index.append(np.random.choice(data_num, int(t_num[i]),replace=False))
    temp_map_list=[]
    for i in range(client_num):
        temp_map={}
        i1=0
        for j in user_data_index[i]:
            temp_map[j]=i1
            i1=i1+1
        temp_map_list.append(temp_map)

    for i in range(client_num):
        t_u_features=[features[index].cuda() for index in user_data_index[i]]
        t_u_labels=[tdata.y[index] for index in user_data_index[i]]
        user_features.append(t_u_features)
        user_labels.append(t_u_labels)
        t=[]
        for index in range(int(t_num[i])):
            if user_data_index[i][index] in train_index:
                t.append(index)
        user_train_index.append(t)

    test_features=[features[index].cuda() for index in test_index]
    val_features=[features[index].cuda() for index in val_index]
    test_features=torch.stack(test_features)
    test_labels=[tdata.y[index] for index in test_index]
    val_features=torch.stack(val_features)
    val_labels=[tdata.y[index] for index in val_index]



    client_adj_lists=[defaultdict(set) for i in range(client_num)]
    adj_num=len(tdata.edge_index[0])
    for i in range(adj_num):
        for client in range(client_num):
            if int(tdata.edge_index[0][i]) in user_data_index[client] and int(tdata.edge_index[1][i]) in user_data_index[client] :
                a =temp_map_list[client][int(tdata.edge_index[0][i])]
                b =temp_map_list[client][int(tdata.edge_index[1][i])]
                client_adj_lists[client][a].add((b,adj[int(tdata.edge_index[0][i])][int(tdata.edge_index[1][i])]))
                #client_adj_lists[client][b].add((a,adj[int(tdata.edge_index[1][i])][int(tdata.edge_index[0][i])]))
    for i in range(client_num):
        t_l=len(client_adj_lists[i])
        for j in range(t_l):
            client_adj_lists[client][j].add((j,0.25))
    return user_features,client_adj_lists,user_labels,label_num,user_train_index,test_features,test_labels,val_features,val_labels
