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
    dataset_str='PubMed'
    #dataset_str='Cora'
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

    #IID 划分（v_deal_idd.py）：
    #  随机划分：节点和边是 随机划分 的，每个客户端获得的数据 分布相似，符合 独立同分布（IID）。
    #  均匀分配：数据和特征被 均匀 地分配给每个客户端，确保每个客户端持有 相似的数据分布。
    n_r=np.random.permutation(data_num) # 随机排列数据的索引
    train_index=n_r[:int(0.7*data_num)].tolist() ## 训练集索引,将前 70% 的数据作为训练集索引
    test_index=n_r[int(0.7*data_num):].tolist() ## 测试集索引,将后 30% 的数据作为测试集索引。这个划分是随机的

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) ## 获取邻接矩阵,将图转换为邻接矩阵（graph 是图的字典表示形式）
    edge_num=adj.shape[0] ## 获取图中的边数（即邻接矩阵的大小）

    #c_e_num=[i*edge_num for i in [0,0.5,1]] #将所有边的数量划分成三个段，分别为 0% 到 50%，50% 到 100%，以及整个边的数量。这是为了后面分配边到不同的客户端。
    #c_n_num=[i*feature_num for i in [0,1,0,1]] # 节点特征数量的分配，将节点特征按比例划分（例如：0到100%，100%到200%等），这样每个客户端就可以分配不同的节点特征。

    #e_t=np.random.permutation(edge_num) # 随机打乱边
    #n_t=np.random.permutation(feature_num)
    #n_t=np.array([i for i in range(feature_num)]) #创建一个包含所有节点特征索引的数组。feature_num 是节点特征的数量，n_t 就是节点特征的索引数组。
    #c_edge_index=[] #用来存储每个客户端所分配到的边的索引。

    #adj_list=[] #用来存储每个客户端的邻接矩阵。
    #for i in range(client_num):
     #   adj1=adj[train_index,:][:,train_index] # 按训练集索引获取子图,基于训练集的索引，从邻接矩阵 adj 中提取出训练集对应的子图。每个客户端将会拥有这个子图的一部分。
      #  c_edge_index.append(np.setdiff1d(e_t, e_t[int(c_e_num[i]):int(c_e_num[i+1])])) # 分配边,通过 np.setdiff1d 将边随机分配给不同的客户端。e_t 被分为多个区间，每个客户端获取其中的一部分边。具体来说，分配的是 e_t 中的边的索引。
       # ee=np.array(adj1.todense()).nonzero()[0][c_edge_index[i]].tolist() # 获取边的连接
       # ee1=np.array(adj1.todense()).nonzero()[1][c_edge_index[i]].tolist() # 获取边的连接,这两行代码获取的是分配给当前客户端的边的 连接节点。nonzero() 用于找出邻接矩阵中非零元素的索引，即每条边连接的节点。
        #for x,y in zip(ee,ee1):#将当前客户端不需要的边置为 0，这样就只保留了属于当前客户端的边。
         #   adj1[(x,y)]=0
    
    # ========== IID边划分：均匀随机分配 ==========
    # 1. 准备边的索引列表
    adj_train = adj[train_index,:][:,train_index]
    all_edges = list(zip(*adj_train.nonzero()))  # 获取所有边的(行,列)对
    total_edges = len(all_edges)
    
    # 2. 随机打乱并均匀分配
    random.shuffle(all_edges)
    edges_per_client = total_edges // client_num
    
    print(f"\n[IID边划分] 总边数: {total_edges}, 每客户端约: {edges_per_client}")
    
    c_edge_index=[]
    adj_list=[]
    
    for i in range(client_num):
        # 3. 为当前客户端分配边
        start_idx = i * edges_per_client
        end_idx = (i + 1) * edges_per_client if i < client_num - 1 else total_edges
        client_edges = set(all_edges[start_idx:end_idx])
        
        print(f"  客户端 {i}: {len(client_edges)} 条边 ({100*len(client_edges)/total_edges:.1f}%)")
        
        # 4. 构建该客户端的邻接矩阵
        adj1 = adj_train.copy()
        adj1 = adj1.tolil()  # 转为LIL格式便于修改
        
        # 5. 删除不属于该客户端的边
        rows, cols = adj1.nonzero()
        for r_idx, c_idx in zip(rows, cols):
            if (r_idx, c_idx) not in client_edges:
                adj1[r_idx, c_idx] = 0
        
        c_edge_index.append(list(client_edges))

        #对邻接矩阵进行 对称性调整（因为图是无向的）。然后，使用 AugNormAdj 对邻接矩阵进行规范化，并将其转换为 PyTorch 稀疏矩阵，最后存储到 adj_list 中。
        adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
        adj1=adj1.todense()
        adj_sp=sp.csr_matrix(adj1)
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj2 = adj_normalizer(adj_sp) 
        adj3 = sparse_mx_to_torch_sparse_tensor(adj2)
        adj4=get_A_r(adj3,1)
        adj_list.append((adj4).cuda())
#self
    #对于每个客户端，分配 训练集特征 和 标签，并将其添加到 user_features 和 train_features 中。
    #user_features=[]
    #user_labels=[]
    #train_features=[]
    #train_labels=[]
    #for i in range(client_num):
     #   user_features.append([features[index].cuda() for index in train_index])
        # ↑ 每个client都得到完整的features，没有特征分割！
      #  train_features.append([features[index] for index in train_index])
    
   # train_labels.append([labels[index] for index in train_index])
    #将 测试集特征 按照 节点特征的划分 分配给每个客户端。
   # test_features=[torch.stack([features[index][n_t[int(c_n_num[k]):int(c_n_num[k+1])]].cuda() for index in test_index]) for k in range(0,client_num+1,2)]
   # test_features=torch.stack(test_features)

   # test_labels=[labels[index] for index in test_index]
    #at_features=[torch.stack([features[index][n_t[int(c_n_num[k]):int(c_n_num[k+1])]].cuda() for index in train_index]) for k in range(0,client_num+1,2)]
    #self
    # ========== IID特征划分：垂直维度均分 ==========
    feature_per_client = feature_num // client_num
    print(f"\n[IID特征划分] 总特征维度: {feature_num}, 每客户端约: {feature_per_client}")
    
    user_features=[]
    user_labels=[]
    train_features=[]
    train_labels=[]

     # ⚠️ 计算最大特征维度（用于统一padding）
    max_feat_dim = feature_per_client if feature_num % client_num == 0 else feature_per_client + 1
    
    
    for i in range(client_num):
        # 计算当前客户端的特征维度范围
        feat_start = i * feature_per_client
        feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
        current_dim = feat_end - feat_start
        
        print(f"  客户端 {i}: 特征维度 [{feat_start}:{feat_end}] (共{feat_end-feat_start}维)")

        # ✅ 获取特征切片
        client_user_features = []
        client_train_features = []
        
        for index in train_index:
            feat_slice = features[index][feat_start:feat_end]
            
            # ⚠️ 如果需要padding，添加零填充
            if current_dim < max_feat_dim:
                padding_size = max_feat_dim - current_dim
                padding = torch.zeros(padding_size)
                feat_slice = torch.cat([feat_slice, padding])
            
            client_user_features.append(feat_slice.cuda())
            client_train_features.append(feat_slice)
        
        if current_dim < max_feat_dim:
            print(f" -> padded to {max_feat_dim}维")
        else:
            print()
        
        user_features.append(client_user_features)
        train_features.append(client_train_features)
        
        # ✅ 关键修改：每个客户端只获取自己的特征维度切片
        #user_features.append([features[index][feat_start:feat_end].cuda() for index in train_index])
        #train_features.append([features[index][feat_start:feat_end] for index in train_index])
    
    train_labels.append([labels[index] for index in train_index])
    
    # 测试集特征也按维度划分
    #test_features = []
    #for i in range(client_num):
     #   feat_start = i * feature_per_client
      #  feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
       # test_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in test_index])
       # test_features.append(test_feat)
    #test_features = torch.stack(test_features)
    #test_labels = [labels[index] for index in test_index]

    # 测试集特征也按维度划分 (添加padding处理)
    test_features = []
    max_feat_dim = feature_per_client if feature_num % client_num == 0 else feature_per_client + 1
    
    for i in range(client_num):
        feat_start = i * feature_per_client
        feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
        
        # 获取当前客户端的特征切片
        test_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in test_index])
        
        # 如果当前维度小于最大维度，进行zero-padding
        current_dim = feat_end - feat_start
        if current_dim < max_feat_dim:
            padding_size = max_feat_dim - current_dim
            padding = torch.zeros(test_feat.shape[0], padding_size, device=test_feat.device)
            test_feat = torch.cat([test_feat, padding], dim=1)
        
        test_features.append(test_feat)
    
    test_features = torch.stack(test_features)
    test_labels = [labels[index] for index in test_index]
    
    # at_features同样处理
    #at_features = []
    #for i in range(client_num):
     #   feat_start = i * feature_per_client
      #  feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
       # at_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in train_index])
        #at_features.append(at_feat)
    #at_labels=[labels[index] for index in train_index]
    #at_features=torch.stack(at_features)

   # at_features 同样处理 (添加padding)
    at_features = []
    max_feat_dim = feature_per_client if feature_num % client_num == 0 else feature_per_client + 1
    
    for i in range(client_num):
        feat_start = i * feature_per_client
        feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
        
        # 获取当前客户端的特征切片
        at_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in train_index])
        
        # 如果当前维度小于最大维度，进行zero-padding
        current_dim = feat_end - feat_start
        if current_dim < max_feat_dim:
            padding_size = max_feat_dim - current_dim
            padding = torch.zeros(at_feat.shape[0], padding_size, device=at_feat.device)
            at_feat = torch.cat([at_feat, padding], dim=1)
        
        at_features.append(at_feat)
    
    at_labels = [labels[index] for index in train_index]
    at_features = torch.stack(at_features)

    return user_features,user_labels,adj_list,label_num,train_features,train_labels,test_features,test_labels,at_features,at_labels
