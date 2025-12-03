import argparse
import torch
from deal_data_self import deal_dataset
import numpy as np
from model import Mlp,GMLP
import torch.optim as optim
from tqdm import tqdm
import sklearn
from sklearn.metrics import f1_score
import torch.nn.functional as F
import math
from collections import defaultdict
import collections
import sys
import os

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--client_num', type=int, default='6')
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--round1', type=int, default=30)
parser.add_argument('--round2', type=int, default=20)
parser.add_argument('--epochs1', type=int, default=200)
parser.add_argument('--epochs2', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--batch_size1', type=int, default=30,
                    help='batch size')
parser.add_argument('--outputdim', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--cuda', action='store_true',help='use CUDA')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#check cuda and use it
if torch.cuda.is_available():
	  if not args.cuda:
		    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	  else:
		    device_id = torch.cuda.current_device()
		    print('using device', device_id, torch.cuda.get_device_name(device_id))
device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)
client_num=args.client_num
print("client_num:",client_num)
path = "./dataset/%s" % args.dataSet + "/"
#get data(iid)
user_features,client_adj_lists,user_labels,label_num,user_train_index,test_features,test_labels,val_features,val_labels=deal_dataset(path,args.dataSet,client_num)
feature_dim=len(user_features[0][0])
local_model=[]
local_model1=[]
optimizer=[]
optimizer1=[]
for i in range(client_num):
    local_model.append(Mlp(feature_dim,feature_dim,args.dropout))
    local_model1.append(GMLP(feature_dim,args.outputdim,label_num,args.dropout))
    optimizer.append(optim.Adam(local_model[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
    optimizer1.append(optim.Adam(local_model1[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
global_model1=Mlp(feature_dim,feature_dim,args.dropout)
global_model2=GMLP(feature_dim,args.outputdim,label_num,args.dropout)
if args.cuda:
    for i in range(client_num):
        local_model[i].cuda()
        local_model1[i].cuda()
    global_model2.cuda()
def client_loss(beta,batch_adj,batch_output,batch_input):
    x2_list=[]
    for key in batch_adj:
        for value in batch_adj[key]:
            t=torch.norm(torch.sub(batch_output[key],batch_output[value]).float(),p=2)
            x2_list.append(t*t)
    x2=sum(x2_list)
    loss=beta*x2*0.5
    return loss,x2

def get_batch(client):
    features_num=len(user_features[client])
    rand_indx = torch.tensor(np.random.choice(features_num, args.batch_size,replace=False)).type(torch.long).cuda()
    node_map={}
    for i in range(args.batch_size):
        node_map[int(rand_indx[i])]=i
    features_batch=[]
    for j in rand_indx:
        j=int(j)
        features_batch.append(torch.tensor(np.array(user_features[client][j])).float().cuda())
    adj_batch=defaultdict(set) 
    for index in range(args.batch_size):
        for x in client_adj_lists[client][int(rand_indx[index])]:
            if x in rand_indx:
                adj_batch[index].add(node_map[x])
    features_batch=torch.stack(features_batch)
    return features_batch, adj_batch
def get_batch_label(client):
    features_num=len(user_features[client])
    rand_indx = torch.tensor(np.random.choice(user_train_index[client], args.batch_size1,replace=False)).type(torch.long).cuda()
    features_batch=[]
    for j in rand_indx:
        j=int(j)
        features_batch.append(torch.tensor(np.array(user_features[client][j])).float().cuda())
    labels_batch=torch.tensor(np.array([user_labels[client][int(j)] for j in rand_indx]))
    features_batch=torch.stack(features_batch)
    return features_batch,labels_batch.cuda()
def get_beta(beta,x2):
    de=1e-3
    r_num=len(beta)

    f_b=2.0+sum(x2)/2.0
    for i in range(1,r_num+1):

        last_b=0.0

        t_b=1
        while (abs(beta[i-1]-last_b)>de):
            t=math.sqrt((2.0*np.log(r_num))/(t_b*f_b*f_b))
            f=[2*beta[j]+x2[j] for j in range(r_num)]
            f_sum=sum([beta[j]*math.exp(-t*f[j]) for j in range(r_num)])
            last_b=beta[i-1]
            beta[i-1]=(beta[i-1]*math.exp(-t*f[i-1]))/f_sum
            t_b=t_b+1
    return beta
def get_average(models,global_model):
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)
    global_model.load_state_dict(fed_state_dict)
    return global_model
def upload_local_model(global_model,models):
    for model in models:
        model.load_state_dict(global_model.state_dict())
    return
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels),sklearn.metrics.f1_score(labels, preds, average='weighted')  
def test(model):
    output=model(test_features)
    print(output)
    acc_test,micro = accuracy(output, torch.tensor(test_labels))
    print(acc_test,micro)
    f=open('result.txt','w',encoding='utf-8')
    print(acc_test,micro,f)
    f.close()
    return acc_test
def val(model):
    output=model(val_features)
    acc_test,micro = accuracy(output, torch.tensor(val_labels))
    print(acc_test,micro)
    return acc_test
if __name__ == '__main__':
    beta=[1.0/client_num for i in range(client_num)]
    for r in range(args.round1):
      
        x2=[0.0 for x in range(client_num)]
        for i in range(client_num):
            for t in range(args.epochs1):
                feature_batch,adj_batch=get_batch(i)
                local_model[i].train()
                optimizer[i].zero_grad()
                batch_output=local_model[i](feature_batch)
                loss,x2[i]=client_loss(beta[i],adj_batch,batch_output,feature_batch)
                loss.backward()
                optimizer[i].step()     
        beta=get_beta(beta,x2)
        print(r,beta)
        global_model1=get_average(local_model,global_model1)
        upload_local_model(global_model1,local_model)
    for r in range(args.round2):
        for i in range(client_num):
            for t in range(args.epochs2):
                feature_batch,label_batch=get_batch_label(i)
                local_model1[i].train()
                optimizer1[i].zero_grad()
                batch_output=local_model1[i](feature_batch)
                loss = F.nll_loss(batch_output, label_batch)
                loss.backward()
                optimizer1[i].step()     
        global_model2=get_average(local_model1,global_model2)
        upload_local_model(global_model2,local_model1)
        print(r)
        val(global_model2)
    test(global_model2)
   

