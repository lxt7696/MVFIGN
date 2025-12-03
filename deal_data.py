import sys
import os
import random
import numpy as np
from collections import defaultdict
import torch
#if __name__ == '__main__':

def deal_dataset(path,dataset,client_num):
    
    f_l_file=path+dataset+".content"
    adj_file=path+dataset+".cites"
    feature_data=[]
    labels_data = [] 
    label_map = {} 
    node_map={}
    test_features=[]
    test_adj={}
    test_label=[]
    with open(f_l_file) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feature_data.append([int(x) for x in info[1:-1]])
            node_map[info[0]] = i
            if info[-1] not in label_map:
                label_map[info[-1]] = len(label_map)
            labels_data.append(label_map[info[-1]])
    data_num=len(feature_data)
    feature_num=len(feature_data[0])
    label_num=len(label_map)
    rand_data=np.random.permutation(data_num).tolist()
#6:2:2 train:test:val
    test_index=rand_data[:int(data_num*(0.3))]
    val_index=rand_data[:int(data_num*(0.3))]
    train_index=rand_data[int(data_num*(0.3)):]

    user_data_index=[]
    user_features=[]
    user_labels=[]

    train_data_num=len(train_index)
#random sample [0.7,0.6,...]
    t_num=[i*train_data_num for i in [0.7,0.6,0.5,0.5,0.4,0.3]]
    for i in range(client_num):
        user_data_index.append(np.random.choice(train_index, int(t_num[i]),replace=False))

    for i in range(client_num):
        t_u_features=[feature_data[index] for index in user_data_index[i]]
        t_u_labels=[labels_data[index] for index in user_data_index[i]]
        user_features.append(t_u_features)
        user_labels.append(t_u_labels)

    temp_map_list=[]
    for i in range(client_num):
        temp_map={}
        i1=0
        for j in user_data_index[i]:
            temp_map[j]=i1
            i1=i1+1
        temp_map_list.append(temp_map)
    client_adj_lists=[defaultdict(set) for i in range(client_num)]
    with open(adj_file) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            assert len(info) == 2
            for j in range(client_num):                       
                if info[0] in node_map.keys() and info[1] in node_map.keys() and node_map[info[0]] in user_data_index[j] and node_map[info[1]] in user_data_index[j] :
                        a =temp_map_list[j][node_map[info[0]]]
                        b =temp_map_list[j][node_map[info[1]]]
                        client_adj_lists[j][a].add(b)
                        client_adj_lists[j][b].add(a)
    test_features=[torch.tensor(np.array(feature_data[index])).float().cuda() for index in test_index]
    val_features=[torch.tensor(np.array(feature_data[index])).float().cuda() for index in val_index]
    test_features=torch.stack(test_features)
    test_labels=[labels_data[index] for index in test_index]
    val_features=torch.stack(val_features)
    val_labels=[labels_data[index] for index in val_index]
    return user_features,client_adj_lists,user_labels,label_num,test_features,test_labels,val_features,val_labels
