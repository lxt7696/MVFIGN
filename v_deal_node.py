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
    #dataset_str='PubMed'
    dataset_str='Cora'
    tdataset=Planetoid(root='torch_dataset/',name=dataset_str)
    tdata=tdataset[0]
    labels=tdata['y']
    data_num=len(tdata['x'])
    feature_num=len(tdata['x'][0])
    label_num=len(tdata['y'].unique())
    features=[torch.from_numpy(i) for i in row_normalize(tdata['x'])]
    user_data_index=[]
    #with open("/home/lxt/project/MVFIGN/torch_dataset/PubMed/raw/ind.pubmed.graph", 'rb') as f:
    #with open("/home/lxt/project/MVFIGN/torch_dataset/Citeseer/raw/ind.citeseer.graph", 'rb') as f:
    with open("/home/lxt/project/MVFIGN/torch_dataset/Cora/raw/ind.cora.graph", 'rb') as f:
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
    n_r=np.random.permutation(data_num) # 随机排列数据的索引
    train_index=n_r[:int(0.7*data_num)].tolist() ## 训练集索引,将前 70% 的数据作为训练集索引
    test_index=n_r[int(0.7*data_num):].tolist() ## 测试集索引,将后 30% 的数据作为测试集索引。这个划分是随机的

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) ## 获取邻接矩阵,将图转换为邻接矩阵（graph 是图的字典表示形式）
    edge_num=adj.shape[0] ## 获取图中的边数（即邻接矩阵的大小）

    # ========== 按节点标签划分边 (论文5.4.3方法2) ==========
    #c_n_num=[i*feature_num for i in [0,1,0,1]]
    #n_t=np.array([i for i in range(feature_num)])
    
    # 标签划分重点代码
    # !!!获取训练节点的标签
    #按节点标签划分（v_deal_node.py）：
    #  标签划分：根据节点标签将不同类别的节点分配给不同的客户端，每个客户端持有不同标签的节点和相关的图边。
    #  非独立同分布（Non-IID）：每个客户端拥有 不同标签的数据，因此数据分布 不相同。
    train_labels_array = torch.tensor([labels[i].item() for i in train_index])
    unique_labels = train_labels_array.unique().tolist() #将训练集的标签 提取出来，找到唯一的标签（unique_labels）
    num_labels = len(unique_labels)
    
    # 为每个客户端分配标签类别,使用 标签均匀分配 的策略，将每个标签分配给不同的客户端。labels_per_client 列表用于存储每个客户端持有的标签。
    labels_per_client = [[] for _ in range(client_num)]
    for idx, label in enumerate(unique_labels):
        client_id = idx % client_num
        labels_per_client[client_id].append(label)
    
    #######################################

    print(f"[按标签划分] 客户端数量: {client_num}, 标签类别数: {num_labels}")
    for i, labels_list in enumerate(labels_per_client):
        print(f"  客户端 {i} 分配的标签: {labels_list}")
    
    c_edge_index=[]
    adj_list=[]

    #边划分：对于每个客户端，按标签划分同类边与跨类边。保留同类边，并 随机分配跨类边 给每个客户端。
    #这样，客户端之间的 图结构 也根据 节点标签划分 了
    for i in range(client_num):
        adj1=adj[train_index,:][:,train_index]
        adj1 = adj1.tolil()
        
        # 获取所有边
        rows, cols = adj1.nonzero()
        assigned_labels_set = set(labels_per_client[i])
        
        # 分类边: 同类边(保留) vs 跨类边(待分配)
        same_class_edges = []
        cross_class_edges = []
        
        for edge_idx in range(len(rows)):
            r_idx, c_idx = rows[edge_idx], cols[edge_idx]
            label_r = train_labels_array[r_idx].item()
            label_c = train_labels_array[c_idx].item()
            
            # 两个节点都属于该客户端的标签 -> 保留
            if label_r in assigned_labels_set and label_c in assigned_labels_set:
                same_class_edges.append((r_idx, c_idx))
            else:
                cross_class_edges.append((r_idx, c_idx))
        
        # 随机分配跨类边
        random.shuffle(cross_class_edges)
        edges_per_client = len(cross_class_edges) // client_num
        start_idx = i * edges_per_client
        end_idx = start_idx + edges_per_client if i < client_num - 1 else len(cross_class_edges)
        assigned_cross_edges = cross_class_edges[start_idx:end_idx]
        
        # 保留的边 = 同类边 + 分配的跨类边
        kept_edges_set = set(same_class_edges + assigned_cross_edges)
        
        # 删除不属于该客户端的边
        for r_idx, c_idx in zip(rows, cols):
            if (r_idx, c_idx) not in kept_edges_set:
                adj1[r_idx, c_idx] = 0
        
        print(f"  客户端 {i}: 同类边 {len(same_class_edges)}, 跨类边 {len(assigned_cross_edges)}, 总计 {len(kept_edges_set)}")
        
        c_edge_index.append(list(kept_edges_set))
        
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
    #self
    # ========== 特征垂直划分：每个客户端只持有部分维度 ==========
    feature_per_client = feature_num // client_num
    print(f"\n[特征垂直划分] 总特征维度: {feature_num}, 每客户端约: {feature_per_client}")
    
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
        
        # ✅ 关键修改：每个客户端只获取自己的特征维度切片
        #user_features.append([features[index][feat_start:feat_end].cuda() for index in train_index])
        #train_features.append([features[index][feat_start:feat_end] for index in train_index])
        user_features.append(client_user_features)
        train_features.append(client_train_features)
    
    train_labels.append([labels[index] for index in train_index])
    
    # 测试集特征也按维度划分
    #test_features = []
    #for i in range(client_num):
     #   feat_start = i * feature_per_client
      #  feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
       # test_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in test_index])
        #test_features.append(test_feat)
    #test_features = torch.stack(test_features)
    
   # test_labels = [labels[index] for index in test_index]

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

    # at_features 同样处理
    #at_features = []
    #for i in range(client_num):
     #   feat_start = i * feature_per_client
      #  feat_end = (i + 1) * feature_per_client if i < client_num - 1 else feature_num
       # at_feat = torch.stack([features[index][feat_start:feat_end].cuda() for index in train_index])
        #at_features.append(at_feat)
    
    #at_labels = [labels[index] for index in train_index]
    #at_features = torch.stack(at_features)

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
