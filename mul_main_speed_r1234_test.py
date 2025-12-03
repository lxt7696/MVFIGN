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
import psutil
torch.cuda.set_device(0)

# ==================== 新增：推理性能测试函数 ====================

def count_parameters(model):
    """计算模型参数总数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_inference_performance(model, test_features, test_labels, val_features, val_labels,
                               dataset_name="PubMed", model_name="MIGN",
                               warmup_runs=20, test_runs=100):
    """
    完整的推理性能测试
    包括：准确率、推理时间、参数量、吞吐量
    """
    print("\n" + "="*80)
    print(f"推理性能测试 - {model_name} ({dataset_name})")
    print("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # 1. 计算参数量
    print("\n[1] 模型参数统计:")
    total_params, trainable_params = count_parameters(model)
    print(f"    总参数数量: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"    模型大小: {model_size_mb:.2f} MB")
    
    # 2. 测试准确率（验证集和测试集）
    print("\n[2] 模型准确率:")
    with torch.no_grad():
        val_output = model(val_features)
        val_preds = val_output.max(1)[1].type_as(torch.tensor(val_labels, device=device))
        val_acc = (val_preds.cpu().numpy() == np.array(val_labels)).mean()
        
        test_output = model(test_features)
        test_preds = test_output.max(1)[1].type_as(torch.tensor(test_labels, device=device))
        test_acc = (test_preds.cpu().numpy() == np.array(test_labels)).mean()
    
    print(f"    验证集准确率: {val_acc*100:.2f}%")
    print(f"    测试集准确率: {test_acc*100:.2f}%")
    
    # 3. 预热
    print(f"\n[3] GPU预热 ({warmup_runs} 次)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(test_features)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # 4. 测试推理时间
    print(f"\n[4] 测试推理时间 ({test_runs} 次)...")
    inference_times = []
    
    with torch.no_grad():
        for _ in range(test_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(test_features)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # 转为毫秒
    
    # 5. 计算统计信息
    avg_time_ms = np.mean(inference_times)
    std_time_ms = np.std(inference_times)
    median_time_ms = np.median(inference_times)
    min_time_ms = np.min(inference_times)
    max_time_ms = np.max(inference_times)
    
    # 6. 计算吞吐量
    num_samples = test_features.shape[0]
    throughput = (num_samples / avg_time_ms) * 1000  # samples/sec
    
    # 7. 输出结果
    print("\n" + "="*80)
    print("推理性能测试结果")
    print("="*80)
    print(f"\n【准确率】")
    print(f"  验证集: {val_acc*100:.2f}%")
    print(f"  测试集: {test_acc*100:.2f}%")
    
    print(f"\n【模型参数】")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    
    print(f"\n【推理时间】(样本数: {num_samples})")
    print(f"  平均时间: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
    print(f"  中位数时间: {median_time_ms:.4f} ms")
    print(f"  最小时间: {min_time_ms:.4f} ms")
    print(f"  最大时间: {max_time_ms:.4f} ms")
    
    print(f"\n【吞吐量】")
    print(f"  {throughput:.2f} samples/sec")
    print(f"  {throughput/1000:.4f} K samples/sec")
    
    print("="*80 + "\n")
    
    return {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'avg_inference_time_ms': avg_time_ms,
        'std_inference_time_ms': std_time_ms,
        'throughput': throughput,
    }

def test_memory_usage(model, test_features, device="cuda"):
    """测试GPU/CPU内存占用"""
    print("\n" + "="*80)
    print("[TEST 1] 内存占用分析")
    print("="*80)
    
    model.eval()
    process = psutil.Process(os.getpid())
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    print("\n推理前:")
    if device == "cuda":
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU内存: {gpu_mem_before:.2f} MB")
    cpu_mem_before = process.memory_info().rss / 1024**2
    print(f"  CPU内存: {cpu_mem_before:.2f} MB")
    
    with torch.no_grad():
        _ = model(test_features)
        if device == "cuda":
            torch.cuda.synchronize()
    
    print("\n推理后:")
    if device == "cuda":
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU内存: {gpu_mem_after:.2f} MB")
        print(f"  GPU峰值: {gpu_peak:.2f} MB")
        print(f"  GPU增长: {gpu_mem_after - gpu_mem_before:.2f} MB")
    cpu_mem_after = process.memory_info().rss / 1024**2
    print(f"  CPU内存: {cpu_mem_after:.2f} MB")
    print(f"  CPU增长: {cpu_mem_after - cpu_mem_before:.2f} MB")
    print("="*80)
    
    return gpu_peak if device == "cuda" else None

# ==================== 原有代码 ====================

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--client_num', type=int, default='1')
parser.add_argument('--dataSet', type=str, default='PubMed')
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

parser.add_argument('--test_inference', action='store_true', default=True,
                    help='Whether to test inference performance after training')

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
        PATH2='./test_state_dict2.pth'
        torch.save(local_model[m].state_dict(), PATH2)
        local_model1.tran[m].load_state_dict(torch.load(PATH2))
    for name, param in local_model1.named_parameters():
        if "tran" in name:
            param.requires_grad = False
    tt=0.0
    best_model_state = None  # ← 新增：保存最佳模型状态

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
            # ← 新增：保存最佳模型
            best_model_state = local_model1.state_dict()
            print(f"\nEpoch {epoch4}: 新的最佳模型! 测试准确率: {tt:.4f}")

        log_file1 = open(r"PubMed——loss——r1234——512——test.txt", encoding="utf-8",mode="a+")  
        with log_file1 as file_to_be_write:  
            print(epoch4,loss, tmp_test_acc,file=file_to_be_write)
    log_file = open(r"PubMed——tt——r1234——512——test.txt", encoding="utf-8",mode="a+")  
    with log_file as file_to_be_write:
        print(args.c,args.cm,args.round1,tt, file=file_to_be_write)

    # ← 新增：保存最佳模型到文件
    model_save_path = f"./saved_models/MIGN_PubMed_best_r1234_512_test.pth"
    os.makedirs("./saved_models", exist_ok=True)
    
    torch.save({
        'epoch': epoch4,
        'model_state_dict': best_model_state,
        'test_acc': tt,
        'args': args,
    }, model_save_path)
    
    print(f"\n最佳模型已保存到: {model_save_path}")
    print(f"最佳测试准确率: {tt:.4f}")

    # ==================== 新增：训练完成后测试推理性能 ====================
    if args.test_inference and best_model_state is not None:
        print("\n" + "="*80)
        print("训练完成，开始推理性能测试...")
        print("="*80)
        
        # 加载最佳模型
        local_model1.load_state_dict(best_model_state)
        
        # 测试推理性能
        perf_results = test_inference_performance(
            model=local_model1,
            test_features=test_features,
            test_labels=test_labels,
            val_features=val_features,
            val_labels=val_labels,
            dataset_name=args.dataSet,
            model_name=f"MIGN (r={args.r}, hidden={args.hidden})",
            warmup_runs=10,
            test_runs=100
        )
        
        # 保存推理性能结果到文件
        perf_log_file = f"MIGN_inference_performance_{args.dataSet}_r{args.r}_{args.hidden}.txt"
        with open(perf_log_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"推理性能测试报告 - {args.dataSet} Dataset\n")
            f.write("="*80 + "\n\n")
            
            f.write("【训练配置】\n")
            f.write(f"  数据集: {args.dataSet}\n")
            f.write(f"  Hop数: {args.r}\n")
            f.write(f"  隐层维度: {args.hidden}\n")
            f.write(f"  Dropout: {args.dropout}\n")
            f.write(f"  学习率: {args.lr}\n\n")
            
            f.write("【模型性能】\n")
            f.write(f"  测试集准确率: {perf_results['test_acc']*100:.2f}%\n")
            f.write(f"  验证集准确率: {perf_results['val_acc']*100:.2f}%\n\n")
            
            f.write("【模型参数】\n")
            f.write(f"  总参数数: {perf_results['total_params']:,}\n")
            f.write(f"  可训练参数: {perf_results['trainable_params']:,}\n")
            f.write(f"  模型大小: {perf_results['model_size_mb']:.2f} MB\n\n")
            
            f.write("【推理速度】\n")
            f.write(f"  平均时间: {perf_results['avg_inference_time_ms']:.4f} ± {perf_results['std_inference_time_ms']:.4f} ms\n")
            f.write(f"  吞吐量: {perf_results['throughput']:.2f} samples/sec\n")
            f.write(f"  吞吐量: {perf_results['throughput']/1000:.4f} K samples/sec\n\n")
        
        print(f"推理性能结果已保存到: {perf_log_file}")

     # TEST 1: 内存占用
    test_memory_usage(local_model1, test_features, device="cuda")
   