# v_main_mpc_he.py - 集成MPC和HE的主程序

import argparse
import torch
from v_deal import deal_dataset
import numpy as np
from v_model_mpche_time_improve import SecureTr, SecureGMLP, Tr, GMLP
import torch.optim as optim
from tqdm import tqdm
import sklearn
from sklearn.metrics import f1_score
import torch.nn.functional as F

torch.cuda.set_device(0)

# ⭐ 添加：时间统计类
class TimeStatistics:
    """时间统计类 - 用于记录MPC和HE的时间占比"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_time = 0.0
        self.mpc_time = 0.0
        self.he_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.other_time = 0.0
        self.count = 0
    
    def add_record(self, mpc_t, he_t, forward_t, backward_t, other_t):
        self.mpc_time += mpc_t
        self.he_time += he_t
        self.forward_time += forward_t
        self.backward_time += backward_t
        self.other_time += other_t
        self.total_time += (mpc_t + he_t + forward_t + backward_t + other_t)
        self.count += 1
    
    def print_summary(self, phase="Training"):
        if self.total_time == 0:
            print(f"\n[{phase}] No time data recorded")
            return
        
        print(f"\n{'='*70}")
        print(f"{phase} Time Statistics Summary")
        print(f"{'='*70}")
        print(f"Total Iterations: {self.count}")
        print(f"Total Time: {self.total_time:.2f}s")
        print(f"{'-'*70}")
        print(f"{'Component':<20} {'Time (s)':<15} {'Percentage':<15}")
        print(f"{'-'*70}")
        print(f"{'MPC Time':<20} {self.mpc_time:<15.2f} {self.mpc_time/self.total_time*100:<15.1f}%")
        print(f"{'HE Time':<20} {self.he_time:<15.2f} {self.he_time/self.total_time*100:<15.1f}%")
        print(f"{'Forward Time':<20} {self.forward_time:<15.2f} {self.forward_time/self.total_time*100:<15.1f}%")
        print(f"{'Backward Time':<20} {self.backward_time:<15.2f} {self.backward_time/self.total_time*100:<15.1f}%")
        print(f"{'Other Time':<20} {self.other_time:<15.2f} {self.other_time/self.total_time*100:<15.1f}%")
        print(f"{'='*70}")
        print(f"MPC + HE Total: {(self.mpc_time+self.he_time)/self.total_time*100:.1f}%")
        print(f"{'='*70}\n")

parser = argparse.ArgumentParser(description='pytorch version of TEE-VFL with MPC and HE')


# 原有参数
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
parser.add_argument('--c', type=int, default=2)
parser.add_argument('--d', type=int, default=2)
parser.add_argument('--cm', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay1', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--r', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='use CUDA')

# MPC 相关参数
parser.add_argument('--use_mpc', action='store_true', default=False,
                    help='Enable MPC for secure computation')
parser.add_argument('--party_id', type=int, default=0, choices=[0, 1],
                    help='Party ID for MPC (0 or 1)')
parser.add_argument('--mpc_host', type=str, default='localhost',
                    help='Host address for MPC communication')
parser.add_argument('--mpc_port', type=int, default=30000,
                    help='Base port for MPC communication')
parser.add_argument('--mpc_batch_size', type=int, default=2048,
                    help='Batch size for MPC computation (rows per batch)')

# HE 相关参数
parser.add_argument('--use_he', action='store_true', default=False,
                    help='Enable HE for secure gradient aggregation')
parser.add_argument('--he_freq', type=int, default=1,
                    help='Apply HE aggregation every N rounds in Phase 2')

args = parser.parse_args()

# CUDA 设置
args.cuda = not args.no_cuda and torch.cuda.is_available()
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

# MPC 设置
if args.use_mpc:
    print(f"\n{'='*60}")
    print(f"MPC ENABLED - Running as Party {args.party_id}")
    print(f"  Host: {args.mpc_host}")
    print(f"  Port: {args.mpc_port}")
    print(f"  Batch Size: {args.mpc_batch_size} rows/batch")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print("MPC DISABLED - Running in plain mode")
    print(f"{'='*60}\n")

# HE 设置
if args.use_he:
    print(f"\n{'='*60}")
    print(f"HE ENABLED - Gradient aggregation with homomorphic encryption")
    print(f"  Aggregation Frequency: Every {args.he_freq} rounds")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print("HE DISABLED - Standard gradient aggregation")
    print(f"{'='*60}\n")

client_num = args.client_num
print("client_num:", client_num)

path = "./dataset/%s" % args.dataSet + "/"

# 获取数据
print("Loading dataset...")
user_features, user_labels, client_adj, label_num, train_features, train_labels, \
    test_features, test_labels, at_f, at_l = deal_dataset(client_num, args.rate, args.r)
print(f"Dataset loaded: {len(user_features[0])} training samples")

# 初始化模型
local_model = []
local_model1 = []
optimizer = []
optimizer1 = []

feature_dim = [len(user_features[j][0]) for j in range(client_num)]
print(f"Feature dimensions: {feature_dim}")

# ⭐ 修改点1: 判断是否需要使用安全模型(MPC或HE)
use_secure_model = args.use_mpc or args.use_he

for i in range(client_num):
    # ⭐ 只要MPC或HE任一启用,就使用Secure模型
    if use_secure_model:
        # 使用安全模型(可以只用HE,不用MPC)
        local_model.append(SecureTr(feature_dim[i], args.dropout, 
                                     party_id=args.party_id, 
                                     use_mpc=args.use_mpc,  # ⭐ 传递实际的use_mpc值
                                     batch_size=args.mpc_batch_size))
        local_model1.append(SecureGMLP(r=args.client_num, nfeat=feature_dim,
                                       nhid=args.hidden, nclass=label_num,
                                       dropout=args.dropout, drate=args.cm,
                                       party_id=args.party_id, 
                                       use_mpc=args.use_mpc,  # ⭐ 传递实际的use_mpc值
                                       use_he=args.use_he,
                                       batch_size=args.mpc_batch_size))
    else:
        # 使用原始模型(既不用MPC也不用HE)
        local_model.append(Tr(feature_dim[i], args.dropout))
        local_model1.append(GMLP(r=args.client_num, nfeat=feature_dim,
                                 nhid=args.hidden, nclass=label_num,
                                 dropout=args.dropout, drate=args.cm))
    
    optimizer.append(optim.Adam(local_model[i].parameters(),
                                lr=args.lr, weight_decay=args.weight_decay))
    optimizer1.append(optim.Adam(local_model1[i].parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay1))

if args.cuda:
    for i in range(client_num):
        local_model[i].cuda()
        local_model1[i].cuda()

print("Models initialized")


def client_loss(alpha, beta, adj_batch, batch_output, x1):
    """计算客户端损失"""
    n = len(batch_output)
    ba_ex = torch.unsqueeze(batch_output.cpu(), 1)
    ba_ex1 = ba_ex.repeat(1, n, 1)
    t = torch.norm(torch.sub(ba_ex1, batch_output.cpu()), p=2, dim=2)**2
    pos_t = torch.mul(adj_batch, t.cuda())
    pos_sum = torch.sum(pos_t, dim=1)
    loss = alpha*(x1*x1)/n + beta*(pos_sum.mean())
    return loss, (x1*x1)/n, pos_sum.mean()


def upload_local_model(global_model, models):
    """上传本地模型"""
    for model in models:
        model.load_state_dict(global_model.state_dict())
    return


def accuracy(output, labels):
    """计算准确率"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), sklearn.metrics.f1_score(labels, preds, average='weighted')


def test(model):
    """测试模型"""
    import time  # 添加
    model.eval()
    # 开始计时
    inference_start = time.time()

    with torch.no_grad():
        # ⭐ 修改点2: 判断是否是SecureGMLP(而非判断use_mpc)
        if isinstance(model, (SecureGMLP, GMLP)):# 添加GMLP判断
            #output = model(test_features, compute_norm=False)
            output, mpc_time = model(test_features, compute_norm=False) if isinstance(model, SecureGMLP) else model(test_features)
        else:
            output = model(test_features)
            mpc_time = 0.0
    
    # 结束计时
    inference_end = time.time()
    total_inference_time = inference_end - inference_start

    acc_test, micro = accuracy(output, torch.tensor(test_labels))
    # 打印时间统计
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {acc_test:.4f}, F1: {micro:.4f}")
    print(f"Inference Time: {total_inference_time:.4f}s")
    if mpc_time > 0:
        print(f"  MPC Time: {mpc_time:.4f}s ({mpc_time/total_inference_time*100:.2f}%)")
        print(f"  Other Time: {total_inference_time - mpc_time:.4f}s ({(total_inference_time-mpc_time)/total_inference_time*100:.2f}%)")
    print(f"{'='*60}\n")
    return acc_test


def get_batch1(client):
    """获取批次数据"""
    if client_adj[client].shape[0] > args.batch_size:
        rand_indx = torch.tensor(
            np.random.choice(np.arange(client_adj[client].shape[0]), 
                           args.batch_size)
        ).type(torch.long).cuda()
        features_batch = torch.stack(user_features[client])[rand_indx]
        adj_batch = client_adj[client][rand_indx, :][:, rand_indx]
        return features_batch, adj_batch
    else:
        return torch.stack(user_features[client]), client_adj[client]


def collect_gradients(model):
    """收集模型的梯度"""
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients


def apply_gradients(model, aggregated_grads):
    """应用聚合后的梯度"""
    for name, param in model.named_parameters():
        if name in aggregated_grads and aggregated_grads[name] is not None:
            param.grad = aggregated_grads[name]


if __name__ == '__main__':

    import time
    phase1_stats = TimeStatistics()
    phase2_stats = TimeStatistics()

    # 设置 alpha 和 beta
    if args.c == 1:
        alpha = [1.0/client_num for i in range(client_num)]
        beta = [1.0/client_num for i in range(client_num)]
    elif args.c == 2:
        alpha = [1.0 for i in range(client_num)]
        beta = [1.0 for i in range(client_num)]
    
    print(f"\n{'='*60}")
    print("PHASE 1: Training Local Models")
    print(f"{'='*60}\n")
    
    # Phase 1: 训练本地模型(不使用HE)
    for m in range(client_num):
        print(f"\nTraining local model for client {m}...")
        
        for r in tqdm(range(args.round1), desc=f"Client {m}"):
            round_start = time.time()
            x1 = [0.0 for x in range(client_num)]
            x2 = [0.0 for x in range(client_num)]
            
            f_b, a_b = get_batch1(m)
            
            local_model[m].train()
            optimizer[m].zero_grad()
            
            # 前向传播
            forward_start = time.time()
            # ⭐ 修改点3: 判断是否是SecureTr(而非判断use_mpc)
            if isinstance(local_model[m], SecureTr):
                # SecureTr有compute_norm参数
                # 如果use_mpc=True,才真正计算norm;否则跳过MPC计算
                batch_output, fx1, mpc_time= local_model[m](f_b, compute_norm=args.use_mpc)
            else:
                # 普通Tr没有compute_norm参数
                batch_output, fx1, mpc_time = local_model[m](f_b)
                mpc_time = 0.0
            
            forward_time = time.time() - forward_start - mpc_time
            
            # 计算损失
            loss_start = time.time()
            loss, x1[m], x2[m] = client_loss(alpha[m], beta[m], a_b, 
                                            batch_output, fx1)
            loss_time = time.time() - loss_start
            
            # 反向传播
            backward_start = time.time()
            loss.backward(retain_graph=True)
            backward_time = time.time() - backward_start

            # 优化器步骤
            optim_start = time.time()
            optimizer[m].step()
            optim_time = time.time() - optim_start

            # ⭐ 添加：记录时间
            phase1_stats.add_record(
                mpc_t=mpc_time, # 来自SecureTr中的MPC时间（如果use_mpc=True
                he_t=0.0,  # Phase1不用HE
                forward_t=forward_time,
                backward_t=backward_time,
                other_t=loss_time + optim_time
            )
        
        # 保存模型
        local_model[m].eval()
        PATH2 = f'./model_party{args.party_id}_client{m}.pth'
        torch.save(local_model[m].state_dict(), PATH2)
        print(f"Saved model to {PATH2}")
        
        # 加载到全局模型
        for i in range(client_num):
            # ⭐ 修改点4: 无论是SecureGMLP还是GMLP,都有tran属性
            local_model1[i].tran[m].load_state_dict(torch.load(PATH2))
        
        # ⭐ 添加：打印Phase 1统计
        phase1_stats.print_summary("Phase 1: Local Model Training")
    
    print(f"\n{'='*60}")
    print("PHASE 2: Freezing Local Models & Training Classifier")
    if args.use_he:
        print(f"HE aggregation every {args.he_freq} rounds")
    print(f"{'='*60}\n")
    
    # Phase 2: 冻结 tran 层,训练分类器(使用HE聚合梯度)
    for client_model in local_model1:
        for name, param in client_model.named_parameters():
            if "tran" in name:
                param.requires_grad = False
    
    tt = 0.0
    best_epoch = 0
    
    print("\nTraining classifier...")
    # 用于累计HE时间
    total_he_time = 0.0
    he_call_count = 0

    for epoch4 in tqdm(range(args.round2), desc="Training"):
         # ⭐ 添加：记录本轮时间
        epoch_start = time.time()
        epoch_he_time = 0.0

        local_model1[0].train()
        optimizer1[0].zero_grad()
        
        # 前向传播
        forward_start = time.time()
        # ⭐ 修改点5: 判断是否是SecureGMLP(而非判断use_mpc)
        if isinstance(local_model1[0], SecureGMLP):
            #batch_output = local_model1[0](at_f, compute_norm=False)
            batch_output, mpc_time = local_model1[0](at_f, compute_norm=False)  # 接收2个值
        else:
            #batch_output = local_model1[0](at_f)
            batch_output, mpc_time = local_model1[0](at_f)  # 接收2个值
        forward_time = time.time() - forward_start - mpc_time
        
        # 计算损失
        loss_start = time.time()
        #loss = F.nll_loss(batch_output, torch.tensor(at_l).cuda())
        loss = F.nll_loss(batch_output, torch.tensor(at_l).cuda())
        loss_time = time.time() - loss_start
        # 反向传播
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start

        # ⭐ 使用HE聚合梯度(每he_freq轮)
        if args.use_he and (epoch4 + 1) % args.he_freq == 0:
            print(f"\n[Epoch {epoch4+1}] Applying HE gradient aggregation...")
            
            he_start = time.time()
            # 收集当前客户端的梯度(单机模拟:只有一个客户端在训练)
            # ⭐ 修改：模拟多个客户端的梯度
            # 在单机中，从所有client_num个local_model1中收集梯度
            local_grads = []
            #local_grads = [collect_gradients(local_model1[0])]
            for c_idx in range(client_num):
                grads = collect_gradients(local_model1[c_idx])
                local_grads.append(grads)
            
            print(f"[Epoch {epoch4+1}] Collected gradients from {len(local_grads)} simulated clients")
            
            # 使用HE聚合(即使只有一个客户端,也演示HE流程)
            #aggregated_grads, he_time = local_model1[0].aggregate_gradients_he(local_grads)
            aggregated_grads, he_time_dict = local_model1[0].aggregate_gradients_he(local_grads)

            # ⭐ 处理返回的时间信息
            if isinstance(he_time_dict, dict):
                he_total_time = he_time_dict.get('total', 0.0)
                print(f"[Epoch {epoch4+1}] HE Time Detail: Encrypt={he_time_dict.get('encrypt', 0.0):.4f}s, "
                      f"Agg={he_time_dict.get('aggregate', 0.0):.4f}s, "
                      f"Decrypt={he_time_dict.get('decrypt', 0.0):.4f}s")
            else:
                he_total_time = he_time_dict if isinstance(he_time_dict, (int, float)) else 0.0
            
            
            # 累计HE时间
            total_he_time += he_total_time
            he_call_count += 1
            #epoch_he_time = time.time() - he_start
            epoch_he_time = he_total_time  # ⭐ 直接使用HE操作时间，不包含其他操作

            if aggregated_grads is not None:
                # 应用聚合后的梯度
                apply_gradients(local_model1[0], aggregated_grads)
                print(f"[Epoch {epoch4+1}] HE aggregation applied")
                print(f"  - Updated {len(aggregated_grads)} parameters\n")
            else:
                print(f"[Epoch {epoch4+1}] ⚠ Warning: aggregated_grads is None, skipping update\n")


            #if aggregated_grads is not None:
                # 应用聚合后的梯度
             #   apply_gradients(local_model1[0], aggregated_grads)
              #  print(f"[Epoch {epoch4+1}] HE aggregation applied\n")
        
        # 优化器步骤
        optim_start = time.time()
        optimizer1[0].step()
        optim_time = time.time() - optim_start

        # ⭐ 添加：记录时间
        phase2_stats.add_record(
            mpc_t=mpc_time,
            he_t=epoch_he_time,
            forward_t=forward_time,
            backward_t=backward_time,
            other_t=loss_time + optim_time
        )

        # 测试
        if (epoch4 + 1) % 10 == 0:
            tmp_test_acc = test(local_model1[0])
            if tt < tmp_test_acc:
                tt = tmp_test_acc
                best_epoch = epoch4 + 1
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Test Accuracy: {tt:.4f} (at epoch {best_epoch})")
    print(f"{'='*60}\n")
    
    # 保存结果
    log_file = open(r"log_mpc_he.txt", encoding="utf-8", mode="a+")
    with log_file as file_to_be_write:
        print(f"Party{args.party_id} MPC={args.use_mpc} HE={args.use_he} "
              f"c={args.c} cm={args.cm} round1={args.round1} "
              f"mpc_batch={args.mpc_batch_size} he_freq={args.he_freq} "
              f"acc={tt:.4f}", 
              file=file_to_be_write)
    
    # 详细日志
    log_file2 = open(r"results_mpc_he.txt", encoding="utf-8", mode="a+")
    with log_file2 as f:
        print(f"\n{'='*60}", file=f)
        print(f"Experiment Results - Party {args.party_id}", file=f)
        print(f"{'='*60}", file=f)
        print(f"MPC Enabled: {args.use_mpc}", file=f)
        print(f"MPC Batch Size: {args.mpc_batch_size}", file=f)
        print(f"HE Enabled: {args.use_he}", file=f)
        print(f"HE Frequency: {args.he_freq}", file=f)
        print(f"Dataset: {args.dataSet}", file=f)
        print(f"Clients: {client_num}", file=f)
        print(f"Parameters: c={args.c}, cm={args.cm}, round1={args.round1}", file=f)
        print(f"Best Accuracy: {tt:.4f} (epoch {best_epoch})", file=f)
        print(f"{'='*60}\n", file=f)
    
    print(f"\n{'='*60}")
    print("All results saved!")
    print(f"  - log_mpc_he.txt")
    print(f"  - results_mpc_he.txt")
    print(f"  - model_party{args.party_id}_client*.pth")
    print(f"{'='*60}\n")

    print(f"\n{'='*60}")
print("TRAINING COMPLETED")
print(f"{'='*60}")
print(f"Best Test Accuracy: {tt:.4f} (at epoch {best_epoch})")

print(f"  - model_party{args.party_id}_client*.pth")
print(f"{'='*60}\n")

# ⭐ 添加：打印Phase 2统计
phase2_stats.print_summary("Phase 2: Classifier Training")
    
# ⭐ 添加：打印总体统计
print(f"\n{'='*70}")
print("Overall Training Statistics (Phase 1 + Phase 2)")
print(f"{'='*70}")
total_mpc = phase1_stats.mpc_time + phase2_stats.mpc_time
total_he = phase1_stats.he_time + phase2_stats.he_time
total_all = phase1_stats.total_time + phase2_stats.total_time
    
if total_all > 0:
    print(f"Total Time: {total_all:.2f}s")
    print(f"{'-'*70}")
    print(f"Total MPC Time: {total_mpc:.2f}s ({total_mpc/total_all*100:.1f}%)")
    print(f"Total HE Time: {total_he:.2f}s ({total_he/total_all*100:.1f}%)")
    print(f"Total Forward Time: {(phase1_stats.forward_time + phase2_stats.forward_time):.2f}s")
    print(f"Total Backward Time: {(phase1_stats.backward_time + phase2_stats.backward_time):.2f}s")
    print(f"{'-'*70}")
    print(f"MPC + HE Combined: {(total_mpc+total_he)/total_all*100:.1f}% of total time")
    print(f"{'='*70}\n")

print(f"\n{'='*60}")
print("TRAINING COMPLETED")

if args.use_he and he_call_count > 0:
    print(f"\n{'='*60}")
    print(f"HE Aggregation Statistics (Phase 2)")
    print(f"{'='*60}")
    print(f"  Total HE Time: {total_he_time:.4f}s")
    print(f"  HE Calls: {he_call_count}")
    print(f"  Average HE Time per call: {total_he_time/he_call_count:.4f}s")
    print(f"  HE Time as % of Phase 2: {(total_he_time/(phase2_stats.total_time if phase2_stats.total_time > 0 else 1))*100:.1f}%")
    print(f"{'='*60}\n")

# 添加HE时间统计
if args.use_he and he_call_count > 0:
    print(f"\nHE Aggregation Statistics:")
    print(f"  Total HE Time: {total_he_time:.4f}s")
    print(f"  HE Calls: {he_call_count}")
    print(f"  Average HE Time: {total_he_time/he_call_count:.4f}s per call")

print(f"{'='*60}\n")