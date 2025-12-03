import argparse
import torch
from mul_deal2 import deal_dataset
import numpy as np
from mul_model2 import Tr,GMLP
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
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--client_num', type=int, default='2')
parser.add_argument('--dataSet', type=str, default='PubMed')
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--round1', type=int, default=10)
parser.add_argument('--round2', type=int, default=200)
parser.add_argument('--epochs1', type=int, default=1)
parser.add_argument('--epochs2', type=int, default=10)
parser.add_argument('--epochs3', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--c', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--d', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--cm', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='to compute order-th power of adj')
parser.add_argument('--hidden', type=int, default=512,
                    help='to compute order-th power of adj')
parser.add_argument('--r', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--cuda', action='store_true',help='use CUDA')

args = parser.parse_args()
e=args.epochs1
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
user_features,user_labels,client_adj,label_num,train_features,train_labels,test_features,test_labels,val_features,val_labels,at_f,at_l=deal_dataset(client_num,args.rate,args.r)
feature_dim=len(user_features[0][0])
local_model=[]
local_model1=[]
optimizer=[]
optimizer1=[]
for i in range(client_num):
    local_model.append(Tr(feature_dim,args.dropout))
    local_model1.append(GMLP(r=args.client_num,nfeat=feature_dim,
            nhid=args.hidden,
            nclass=label_num,
            dropout=args.dropout,
            drate=args.cm))
    optimizer.append(optim.Adam(local_model[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
    optimizer1.append(optim.Adam(local_model1[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
global_model1=Tr(feature_dim,args.dropout)
global_model2=GMLP(r=args.client_num,nfeat=feature_dim,
            nhid=args.hidden,
            nclass=label_num,
            dropout=args.dropout)
if args.cuda:
    for i in range(client_num):
        local_model[i].cuda()
        local_model1[i].cuda()
    global_model1.cuda()
    global_model2.cuda()
def client_loss(alpha,beta,adj_batch,batch_output,x1):
    n=len(batch_output)
    ba_ex=torch.unsqueeze(batch_output.cpu(),1)
    ba_ex1 =ba_ex.repeat(1, n, 1)
    t=torch.norm(torch.sub(ba_ex1,batch_output.cpu()),p=2,dim=2)**2
    pos_t=torch.mul(adj_batch,t.cuda())
    pos_sum=torch.sum(pos_t,dim=1)
    loss=alpha*((x1*x1)/n)+beta*(pos_sum.mean())
    print((x1*x1)/n,pos_sum.mean())
    return loss,(x1*x1)/n,pos_sum.mean()

def get_alpha_beta(alpha,beta,x1,x2):
    de=1e-3
    r_num=len(alpha)
    f_a=2.0+sum(x1)
    f_b=2.0+sum(x2)
    for i in range(1,r_num+1):
        last_a=0.0
        last_b=0.0
        t_a=1
        t_b=1
        while (abs(beta[i-1]-last_b)>de):
            t=math.sqrt((2.0*np.log(r_num))/(t_b*f_b*f_b))
            f=[2*beta[j]+x2[j] for j in range(r_num)]
            f_sum=sum([beta[j]*math.exp(-t*f[j]) for j in range(r_num)])
            last_b=beta[i-1]
            beta[i-1]=(beta[i-1]*math.exp(-t*f[i-1]))/f_sum
            t_b=t_b+1
        while (abs(alpha[i-1]-last_a)>de):
            t=math.sqrt((2.0*np.log(r_num))/(t_a*f_a*f_a))
            f=[2*alpha[j]+x1[j] for j in range(r_num)]
            f_sum=sum([alpha[j]*math.exp(-t*f[j]) for j in range(r_num)])
            last_a=alpha[i-1]
            alpha[i-1]=(alpha[i-1]*math.exp(-t*f[i-1]))/f_sum
            t_a=t_a+1
    return alpha,beta
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
    model.eval()
    output=model(test_features)
    acc_test,micro = accuracy(output, torch.tensor(test_labels))
    print(acc_test,micro)
    return acc_test
def val(model):
    model.eval()
    output=model(val_features)
    acc_test,micro = accuracy(output, torch.tensor(val_labels))
    print(acc_test,micro)
    return acc_test
def get_out(client,t_model):
    for t in range(args.epochs3):
        t_model.train()
        optimizer3.zero_grad()
        batch_output= t_model(torch.stack(train_features[client]).cuda())
        loss = F.nll_loss(batch_output, torch.tensor(train_labels[client]).cuda())
        loss.backward()
        optimizer3.step()
    t_model.eval()
    test(t_model)
    output=t_model(torch.stack(user_features[client]))
    output1=output.max(1)[1].type_as(torch.tensor(train_labels[client]))
    return output1
def get_batch(client,output,class_num):
    #user_batch_index=torch.nonzero(output==class_num).squeeze()
    user_batch_index=torch.nonzero(torch.tensor(user_labels[client])==class_num).squeeze()
    if len(user_batch_index)>args.batch_size:
        rand_indx = torch.tensor(np.random.choice(user_batch_index, args.batch_size)).type(torch.long).cuda()
        features_batch = torch.stack(user_features[client])[rand_indx]
        adj_batch = client_adj[client][rand_indx,:][:,rand_indx]
        return features_batch, adj_batch
    else:
          return torch.stack(user_features[client])[user_batch_index],client_adj[client][user_batch_index,:][:,user_batch_index]
def get_batch1(client):
    if client_adj[client].shape[0]>args.batch_size:
        rand_indx = torch.tensor(np.random.choice(np.arange(client_adj[client].shape[0]), args.batch_size)).type(torch.long).cuda()
        features_batch = torch.stack(user_features[client])[rand_indx]
        adj_batch = client_adj[client][rand_indx,:][:,rand_indx]
        return features_batch, adj_batch
    else:
        return torch.stack(user_features[client]),client_adj[client]
if __name__ == '__main__':
    if args.c==1:
        alpha=[1.0/client_num for i in range(client_num)]
        beta=[1.0/client_num for i in range(client_num)]
    if args.c==2:
        alpha=[1.0 for i in range(client_num)]
        beta=[1.0 for i in range(client_num)]
    for m in range(client_num):
        for r in range(args.round1):
            x1=[0.0 for x in range(client_num)]
            x2=[0.0 for x in range(client_num)]
            f_b,a_b=get_batch1(i)
            local_model[m].train()
            optimizer[m].zero_grad()
            batch_output,fx1=local_model[m](f_b)
            loss,x1[m],x2[m]=client_loss(alpha[m],beta[m],a_b,batch_output,fx1)
            loss.backward(retain_graph=True)
            optimizer[m].step()   
        global_model1.eval()
        PATH2='./test_state_dict2.pth'
        torch.save(global_model1.state_dict(), PATH2)
        for i in range(client_num):
            local_model1[i].tran[m].load_state_dict(torch.load(PATH2))
        for client_model in local_model1:
            for name, param in client_model.named_parameters():
                print(name)
                if "tran" in name:
                    param.requires_grad = False
    tt=0.0
    for epoch4 in tqdm(range(args.round2)):
        local_model1[0].train()
        optimizer1[0].zero_grad()
        batch_output= local_model1[0](at_f)
        loss = F.nll_loss(batch_output, torch.tensor(at_l).cuda())
        loss.backward()
        optimizer1[0].step()
        val_acc=val(local_model1[0])
        tmp_test_acc= test(local_model1[0])
        if tt<tmp_test_acc:
            tt=tmp_test_acc
    log_file = open(r"log96.txt", encoding="utf-8",mode="a+")  
    with log_file as file_to_be_write:  
        print(args.c,args.d,args.round1,tt, file=file_to_be_write)

   

