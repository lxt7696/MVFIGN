# v_model_mpc.py - 完整版（使用分块 MPC）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch.autograd import Variable
from mpc_utils import MPSPDZManager

class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, c_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, c_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()
        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_x1(x, y):
    """计算 Frobenius 范数（本地计算版本）"""
    aij_matrix = torch.sub(y, x).half()
    x1 = torch.norm(aij_matrix, p='fro') 
    return x1


def get_x1_secure(party_features, other_party_features, weights, mpc_manager, batch_size=100):
    """
    使用分块 MPC 计算范数
    计算 ||(A-B)W||_F
    
    Args:
        party_features: 当前方的特征
        other_party_features: 未使用（保持接口一致）
        weights: 权重矩阵
        mpc_manager: MPC 管理器
        batch_size: 批处理大小
    """
    party_id = mpc_manager.party_id
    
    print(f"\n[get_x1_secure - Party {party_id}] ========== START ==========")
    print(f"[get_x1_secure - Party {party_id}] party_features shape: {party_features.shape}")
    print(f"[get_x1_secure - Party {party_id}] weights shape: {weights.shape}")
    print(f"[get_x1_secure - Party {party_id}] batch_size: {batch_size}")
    
    # 使用分块计算
    if party_id == 0:
        # Party 0: 提供 A 和 W
        print(f"[get_x1_secure - Party {party_id}] Calling secure_matrix_multiply_batched with A and W...")
        result = mpc_manager.secure_matrix_multiply_batched(
            A=party_features,
            W=weights,
            batch_size=batch_size
        )
        print(f"[get_x1_secure - Party {party_id}] Got result shape: {result.shape}")
        x1 = torch.norm(result, p='fro')
        print(f"[get_x1_secure - Party {party_id}] Computed Frobenius norm: {x1.item():.4f}")
        print(f"[get_x1_secure - Party {party_id}] ========== END ==========\n")
        return x1
    else:
        # Party 1: 提供 B
        print(f"[get_x1_secure - Party {party_id}] Calling secure_matrix_multiply_batched with B...")
        result = mpc_manager.secure_matrix_multiply_batched(
            B=party_features,
            batch_size=batch_size
        )
        print(f"[get_x1_secure - Party {party_id}] Got result shape: {result.shape}")
        # KEY FIX: Party 1 returns a gradient-enabled zero to avoid blocking computation graph
        # Use part of result to maintain gradient connection
        dummy = torch.sum(result * 0.0)  # This maintains gradient graph but with value 0
        print(f"[get_x1_secure - Party {party_id}] Returning gradient-enabled zero: {dummy.item()}")
        print(f"[get_x1_secure - Party {party_id}] ========== END ==========\n")
        return dummy

class Tr(nn.Module):
    """原始的 Tr 类（本地计算）"""
    def __init__(self, hid_dim, dropout):
        super(Tr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim)

    def forward(self, x):
        z = x
        x = self.fc1(x)
        y = x
        x1 = get_x1(z, y)
        return x, x1


class SecureTr(nn.Module):
    """带 MPC 的安全 Tr 类"""
    def __init__(self, hid_dim, dropout, party_id, use_mpc=True, batch_size=100):
        super(SecureTr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim)
        self.party_id = party_id
        self.use_mpc = use_mpc
        self.batch_size = batch_size
        
        # 初始化 MPC 管理器
        if use_mpc:
            self.mpc_manager = MPSPDZManager(party_id=party_id)
            print(f"[SecureTr] Party {party_id} initialized with MPC (batch_size={batch_size})")
        else:
            self.mpc_manager = None
            print(f"[SecureTr] Party {party_id} initialized WITHOUT MPC")
    
    def forward(self, x, compute_norm=True):
        """
        前向传播
        
        Args:
            x: 输入特征
            compute_norm: 是否计算范数（训练时需要）
        
        Returns:
            x: 变换后的特征
            x1: 范数值（如果 compute_norm=True）
        """
        z = x
        x = self.fc1(x)
        y = x
        
        if not compute_norm:
            return x, torch.tensor(0.0)
        
        if self.use_mpc and self.training:
            weights = self.fc1.weight.t()
            x1 = get_x1_secure(z, None, weights, self.mpc_manager, batch_size=self.batch_size)
            
            # KEY: get_x1_secure already returns appropriate values for both parties
            # Party 0 gets actual norm, Party 1 gets gradient-enabled zero
            return x, x1
        else:
            x1 = get_x1(z, y)
            return x, x1


class GMLP(nn.Module):
    """原始的 GMLP 类"""
    def __init__(self, r, nfeat, nhid, nclass, dropout, drate):
        super(GMLP, self).__init__()
        self.nfeat = nfeat
        self.sum_l = sum(nfeat)
        self.nhid = nhid
        self.r = r
        
        self.tran = nn.ModuleList([
            Tr(self.nfeat[i], dropout).cuda() 
            for i in range(self.r)
        ])
        self.classifier = Mlp(self.sum_l, self.nhid, nclass, dropout)
        self.leaky_relu = nn.ModuleList([
            nn.LeakyReLU(drate) 
            for i in range(self.r)
        ])
    
    def forward(self, x):
        m = x
        x, _ = self.tran[0](m[0])
        x = self.leaky_relu[0](x)
        
        for i in range(1, self.r):
            y, _ = self.tran[i](m[i])
            x = torch.cat([x, self.leaky_relu[i](y)], dim=1)
        
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits


class SecureGMLP(nn.Module):
    """带 MPC 的安全 GMLP 类"""
    def __init__(self, r, nfeat, nhid, nclass, dropout, drate, 
                 party_id, use_mpc=True, batch_size=100):
        super(SecureGMLP, self).__init__()
        self.nfeat = nfeat
        self.sum_l = sum(nfeat)
        self.nhid = nhid
        self.r = r
        self.party_id = party_id
        self.use_mpc = use_mpc
        self.batch_size = batch_size
        
        # 使用 SecureTr 替代 Tr
        if use_mpc:
            self.tran = nn.ModuleList([
                SecureTr(self.nfeat[i], dropout, party_id, 
                        use_mpc=True, batch_size=batch_size).cuda() 
                for i in range(self.r)
            ])
        else:
            self.tran = nn.ModuleList([
                Tr(self.nfeat[i], dropout).cuda() 
                for i in range(self.r)
            ])
        
        self.classifier = Mlp(self.sum_l, self.nhid, nclass, dropout)
        self.leaky_relu = nn.ModuleList([
            nn.LeakyReLU(drate) 
            for i in range(self.r)
        ])
        
        print(f"[SecureGMLP] Initialized for Party {party_id}, MPC={use_mpc}, batch_size={batch_size}")
    
    def forward(self, x, compute_norm=True):
        """
        前向传播
        
        Args:
            x: 输入特征列表
            compute_norm: 是否计算范数
        
        Returns:
            class_logits: 分类输出
        """
        m = x
        x, _ = self.tran[0](m[0], compute_norm=compute_norm)
        x = self.leaky_relu[0](x)
        
        for i in range(1, self.r):
            y, _ = self.tran[i](m[i], compute_norm=compute_norm)
            x = torch.cat([x, self.leaky_relu[i](y)], dim=1)
        
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits