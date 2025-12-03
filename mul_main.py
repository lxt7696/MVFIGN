import argparse
import torch
from mul_deal_trans import deal_dataset
import numpy as np
from mul_model import Tr,GMLP
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
import time
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--client_num', type=int, default='1')
parser.add_argument('--dataSet', type=str, default='Cora')
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--round1', type=int, default=300)
parser.add_argument('--round2', type=int, default=300)
parser.add_argument('--epochs1', type=int, default=1)
parser.add_argument('--epochs2', type=int, default=10)
parser.add_argument('--epochs3', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--c', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--d', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--cm', type=float, default=1.0,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay1', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='to compute order-th power of adj')
parser.add_argument('--hidden', type=int, default=512,
                    help='to compute order-th power of adj')
parser.add_argument('--r', type=int, default=4,
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
optimizer=[]
for i in range(args.r):
    local_model.append(Tr(feature_dim,args.dropout))
    optimizer.append(optim.Adam(local_model[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
local_model1=GMLP(r=args.r,nfeat=feature_dim,
            nhid=args.hidden,
            nclass=label_num,
            dropout=args.dropout,drate=args.cm)
optimizer1=optim.Adam(local_model1.parameters(),lr=args.lr, weight_decay=args.weight_decay1)
global_model1=Tr(feature_dim,args.dropout)
global_model2=GMLP(r=args.r,nfeat=feature_dim,
            nhid=args.hidden,
            nclass=label_num,
            dropout=args.dropout,drate=args.cm)
if args.cuda:
    for i in range(args.r):
        local_model[i].cuda()
    local_model1.cuda()
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

def upload_local_model(global_model,models):
    for model in models:
        model.load_state_dict(global_model.state_dict())
    return
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels),sklearn.metrics.f1_score(labels, preds, average='weighted')  
def test_time(model):
    model.eval()
    start = time.clock()
    output=model(test_features)
    end= time.clock()
    print(end - start)
    acc_test,micro = accuracy(output, torch.tensor(test_labels))
    print(acc_test,micro)
    return acc_test
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
def get_batch1(client,d):
    if client_adj[d][client].shape[0]>args.batch_size:
        rand_indx = torch.tensor(np.random.choice(np.arange(client_adj[d][client].shape[0]), args.batch_size)).type(torch.long).cuda()
        features_batch = torch.stack(user_features[client])[rand_indx]
        adj_batch = client_adj[d][client][rand_indx,:][:,rand_indx]
        return features_batch, adj_batch.cuda()
    else:
        return torch.stack(user_features[client]),client_adj[d][client].cuda()

def test_inference_time(model, test_features, test_labels, warmup_runs=10, test_runs=100):
    """
    测试模型的推理速度（只测试forward pass）
    
    参数:
        model: 训练好的模型
        test_features: 测试集特征
        test_labels: 测试集标签
        warmup_runs: 预热次数（GPU需要预热）
        test_runs: 正式测试次数
    
    返回:
        avg_time_ms: 平均推理时间（毫秒）
        std_time_ms: 标准差（毫秒）
        accuracy: 测试准确率
    """
    model.eval()  # 设置为评估模式
    
    # 确保数据在GPU上
    if torch.cuda.is_available():
        test_features = test_features.cuda()
        test_labels_tensor = torch.tensor(test_labels).cuda()
        torch.cuda.synchronize()  # 同步GPU
    else:
        test_labels_tensor = torch.tensor(test_labels)
    
    # 1. 预热阶段（GPU需要预热以获得稳定性能）
    print(f"预热中... ({warmup_runs} 次)")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(test_features)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # 2. 正式测试推理时间
    print(f"测试推理速度... ({test_runs} 次)")
    inference_times = []
    
    with torch.no_grad():  # 关键：不计算梯度
        for _ in range(test_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 等待GPU完成之前的操作
                start_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
            
            # 只做前向传播
            output = model(test_features)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 等待GPU完成当前操作
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 3. 计算统计信息
    avg_time_ms = np.mean(inference_times)
    std_time_ms = np.std(inference_times)
    median_time_ms = np.median(inference_times)
    
    # 4. 计算准确率（使用最后一次的输出）
    preds = output.max(1)[1].type_as(test_labels_tensor)
    correct = preds.eq(test_labels_tensor).sum().item()
    accuracy = correct / len(test_labels)
    
    # 5. 打印结果
    print("\n" + "="*60)
    print("推理速度测试结果:")
    print("="*60)
    print(f"平均推理时间: {avg_time_ms:.4f} ms")
    print(f"标准差: {std_time_ms:.4f} ms")
    print(f"中位数时间: {median_time_ms:.4f} ms")
    print(f"最小时间: {min(inference_times):.4f} ms")
    print(f"最大时间: {max(inference_times):.4f} ms")
    print(f"测试准确率: {accuracy*100:.2f}%")
    print("="*60 + "\n")
    
    return avg_time_ms, std_time_ms, accuracy

if __name__ == '__main__':
    alpha=[1.0 for i in range(args.r)]
    beta=[1.0 for i in range(args.r)]
    x1=[0.0 for x in range(args.r)]
    x2=[0.0 for x in range(args.r)]
    # mul_main.py中的关键特征：
    # 1. 单一客户端（虽然参数叫client_num，但实际用法是centralized）
    for m in range(args.r): # 对不同hop的邻接矩阵训练
        for t in range(args.round1):
            f_b,a_b=get_batch1(0,m) # 注意：固定使用client=0
            local_model[m].train()
            optimizer[m].zero_grad()
            batch_output,fx1=local_model[m](f_b)
            loss,x1[m],x2[m]=client_loss(alpha[m],beta[m],a_b,batch_output,fx1)
            loss.backward(retain_graph=True)
            optimizer[m].step()   
        PATH2='./test_state_dict3.pth'
        torch.save(local_model[m].state_dict(), PATH2)
        local_model1.tran[m].load_state_dict(torch.load(PATH2))
    for name, param in local_model1.named_parameters():
        if "tran" in name:
            param.requires_grad = False
    tt=0.0
    for epoch4 in tqdm(range(args.round2)):
        local_model1.train()
        optimizer1.zero_grad()
        batch_output= local_model1(at_f)
        loss = F.nll_loss(batch_output, torch.tensor(at_l).cuda())
        loss.backward()
        optimizer1.step()
        val_acc=val(local_model1)
        tmp_test_acc= test(local_model1)
        if tt<tmp_test_acc:
            tt=tmp_test_acc
        log_file1 = open(r"Cora——loss——r1234.txt", encoding="utf-8",mode="a+")  
        with log_file1 as file_to_be_write:  
            print(epoch4,loss, tmp_test_acc,file=file_to_be_write)
    log_file = open(r"Cora——tt——r1234.txt.txt", encoding="utf-8",mode="a+")  
    with log_file as file_to_be_write:
        print(args.c,args.cm,args.round1,tt, file=file_to_be_write)

     # 测试最终模型的推理速度
    print("\n" + "="*60)
    print("训练完成，开始测试推理速度")
    print("="*60)
    
    # 直接调用上面定义的test_inference_time函数
    avg_time, std_time, test_acc = test_inference_time(
        model=local_model1,
        test_features=test_features,
        test_labels=test_labels,
        warmup_runs=20,
        test_runs=100
    )

    log_file = open(r"Cora——speed——r1234.txt", encoding="utf-8",mode="a+")  
    with log_file as file_to_be_write:  
        print(avg_time,std_time,test_acc, file=file_to_be_write)



   

