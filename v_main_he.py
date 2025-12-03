import argparse
import torch
from v_deal_mpche import deal_dataset
import numpy as np
from v_model import Tr,GMLP
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
from crypto_utils import BFVEncryption, MPCBeaver  # 新增这一行
import tenseal as ts  # 新增这一行
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--client_num', type=int, default='2')
parser.add_argument('--dataSet', type=str, default='Cora')
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--round1', type=int, default=10)
parser.add_argument('--round2', type=int, default=300)
parser.add_argument('--epochs1', type=int, default=1)
parser.add_argument('--epochs2', type=int, default=10)
parser.add_argument('--epochs3', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--c', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--d', type=int, default=2,
                    help='learning rate.')
parser.add_argument('--cm', type=float, default=1,
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
parser.add_argument('--hidden', type=int, default=256,
                    help='to compute order-th power of adj')
parser.add_argument('--r', type=int, default=3,
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
#user_features,user_labels,client_adj,label_num,train_features,train_labels,test_features,test_labels,at_f,at_l=deal_dataset(client_num,args.rate,args.r)
#get data(iid)
user_features,user_labels,client_adj,label_num,train_features,train_labels,test_features,test_labels,at_f,at_l,mpc_beaver=deal_dataset(client_num,args.rate,args.r)
local_model=[]
local_model1=[]
optimizer=[]
optimizer1=[]
feature_dim=[len(user_features[j][0]) for j in range(client_num)]
for i in range(client_num):
    local_model.append(Tr(feature_dim[i],args.dropout))
    local_model1.append(GMLP(r=args.client_num,nfeat=feature_dim,
            nhid=args.hidden,
            nclass=label_num,
            dropout=args.dropout,drate=args.cm))
    optimizer.append(optim.Adam(local_model[i].parameters(),lr=args.lr, weight_decay=args.weight_decay))
    optimizer1.append(optim.Adam(local_model1[i].parameters(),lr=args.lr, weight_decay=args.weight_decay1))
# ========== 新增: 初始化加密方案 ==========
print("Initializing BFV encryption scheme...")
bfv_enc = BFVEncryption()
public_contexts = [bfv_enc.get_public_context() for _ in range(client_num)]
print("Encryption initialization completed.")
# ==========================================
if args.cuda:
    for i in range(client_num):
        local_model[i].cuda()
        local_model1[i].cuda()
def client_loss(alpha,beta,adj_batch,batch_output,x1):
    n=len(batch_output)
    ba_ex=torch.unsqueeze(batch_output.cpu(),1)
    ba_ex1 =ba_ex.repeat(1, n, 1)
    t=torch.norm(torch.sub(ba_ex1,batch_output.cpu()),p=2,dim=2)**2
    pos_t=torch.mul(adj_batch,t.cuda())
    pos_sum=torch.sum(pos_t,dim=1)
    loss=alpha*(x1*x1)/n+beta*(pos_sum.mean())
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
    #for m in range(client_num):
     #   for r in range(args.round1):
     #       x1=[0.0 for x in range(client_num)]
      #      x2=[0.0 for x in range(client_num)]
       #     f_b,a_b=get_batch1(m)
        #    local_model[m].train()
         #   optimizer[m].zero_grad()
          #  batch_output,fx1=local_model[m](f_b)
           # loss,x1[m],x2[m]=client_loss(alpha[m],beta[m],a_b,batch_output,fx1)
     #       loss.backward(retain_graph=True)
      #      optimizer[m].step()   
       # local_model[m].eval()
        #PATH2='./test_state_dict2.pth'
      #  torch.save(local_model[m].state_dict(), PATH2) # 明文保存
       # for i in range(client_num):
        #    local_model1[i].tran[m].load_state_dict(torch.load(PATH2)) # 明文加载

    # ========== Phase 1: 局部训练 + 加密传输 ==========
    print("\n" + "="*60)
    print("Phase 1: Local Training with Encrypted Parameter Transmission")
    print("="*60)

    phase1_start = time.time()
    
    for m in range(client_num):
        print(f"\n--- Training Client {m} ---")
        
        # 步骤1: 局部训练（不使用MPC，保持梯度正常传播）
        for r in range(args.round1):
            x1=[0.0 for x in range(client_num)]
            x2=[0.0 for x in range(client_num)]
            f_b, a_b = get_batch1(m)
            local_model[m].train()
            optimizer[m].zero_grad()
            
            # 正常的前向传播（不使用MPC）
            batch_output, fx1 = local_model[m](f_b)
            loss, x1[m], x2[m] = client_loss(alpha[m], beta[m], a_b, batch_output, fx1)
            
            # 正常的反向传播
            loss.backward(retain_graph=True)
            optimizer[m].step()
            
            if r % 10 == 0:
                print(f"  Round {r}/{args.round1}, Loss: {loss.item():.4f}")
        
        local_model[m].eval()
        
        # ========== 步骤2: 使用同态加密传输模型参数 ==========
        print(f"\n[Client {m}] Encrypting model parameters with BFV...")
        enc_start = time.time()
        encrypted_state = {}
        
        # 加密每个参数
        for name, param in local_model[m].state_dict().items():
            enc_param, shape = bfv_enc.encrypt_tensor(param)
            encrypted_state[name] = (enc_param, shape)
            print(f"  Encrypted parameter '{name}' with shape {shape}")
        enc_time = time.time() - enc_start

        print(f"[Client {m}] Encryption completed. Total: {len(encrypted_state)} parameters.")
        
        # ========== 步骤3: 模拟服务器聚合（实际应该从所有客户端收集）==========
        print(f"\n[Server] Aggregating encrypted parameters from client {m}...")
        aggregated_enc_state = encrypted_state  # 简化: 单客户端情况
        
        # ========== 步骤4: 解密并分发到全局模型 ==========
        print(f"[Clients] Receiving and decrypting aggregated parameters...")
        dec_start = time.time()
        for i in range(client_num):
            for name in local_model1[i].tran[m].state_dict().keys():
                if name in aggregated_enc_state:
                    enc_param, shape = aggregated_enc_state[name]
                    
                    # 解密参数
                    decrypted_param = bfv_enc.decrypt_tensor(enc_param, shape)
                    
                    # 加载到全局模型
                    local_model1[i].tran[m].state_dict()[name].copy_(decrypted_param.cuda())
        
        dec_time = time.time() - dec_start
        print(f"[Clients] Decryption time: {dec_time:.2f}s")
        print(f"[Client {m}] Parameters distributed securely to all clients.\n")
    
    phase1_time = time.time() - phase1_start
    print("="*60)
    print(f"Phase 1 completed in {phase1_time:.2f}s (with HE encryption)")
    print("Phase 1 Completed: All clients trained with encrypted parameter sharing")
    print("="*60 + "\n")
    # ============================================================
    
    print("\n" + "="*60)
    print("Phase 2: Global Model Training")
    print("="*60)

    for client_model in local_model1:
        for name, param in client_model.named_parameters():
            print(name)
            if "tran" in name:
                param.requires_grad = False
    
    phase2_start = time.time() 
    tt=0.0
    for epoch4 in tqdm(range(args.round2)):
        local_model1[0].train()
        optimizer1[0].zero_grad()
        batch_output= local_model1[0](at_f)
        loss = F.nll_loss(batch_output, torch.tensor(at_l).cuda())
        loss.backward()
        optimizer1[0].step() #直接明文梯度更新，没有HE加密?
        tmp_test_acc= test(local_model1[0])
        if tt<tmp_test_acc:
            tt=tmp_test_acc

    phase2_time = time.time() - phase2_start  # 新增
    total_time = phase1_time + phase2_time

    # ========== 保存结果 ==========
    log_file = open(r"log166.txt", encoding="utf-8",mode="a+")  
    with log_file as file_to_be_write:  
        print(args.c,args.cm,args.round1,tt, file=file_to_be_write)

    log_file = open(r"ablation_mpche.txt", encoding="utf-8",mode="a+")  
    with log_file as f:
        print(f"Ablation: ", file=f)
        print(f"Dataset: {args.dataSet}, Clients: {client_num}, r: {args.r}", file=f)
        print(f"Phase1: {phase1_time:.2f}s, Phase2: {phase2_time:.2f}s, Total: {total_time:.2f}s", file=f)
        print(f"Accuracy: {tt:.4f}", file=f)
        print(f"c={args.c}, cm={args.cm}, round1={args.round1}", file=f)
        print("-"*60, file=f)
    
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS:")
    print(f"  Phase 1 Time: {phase1_time:.2f}s")
    print(f"  Phase 2 Time: {phase2_time:.2f}s")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Best Accuracy: {tt:.4f}")
    print(f"{'='*60}\n")

   

