# v_model_mpc_he.py - 集成MPC和HE的模型文件

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch.autograd import Variable
from mpc_utils import MPSPDZManager
from he_utils_time import GradientAggregator

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


def get_x1_secure(party_features, other_party_features, weights, mpc_manager, batch_size=500):
    """
    使用 MPC 计算范数
    计算 ||(A-B)W||_F
    ⭐ 修改：MPC现在直接返回范数标量，不需要再计算
    """
    import time

    party_id = mpc_manager.party_id
    
    if party_id == 0:
        mpc_start = time.time()
       # ⭐ 使用batched版本
        norm_scalar = mpc_manager.secure_matrix_multiply_batched(
            A=party_features,
            W=weights,
            batch_size=batch_size,
        )
        mpc_time = time.time() - mpc_start
        
        # ⭐ 修改：不需要再调用torch.norm，直接使用返回的标量
        x1 = norm_scalar if norm_scalar is not None else torch.tensor(0.0)
        return x1, mpc_time
    else:
        mpc_start = time.time()
        norm_scalar = mpc_manager.secure_matrix_multiply_batched(
            B=party_features,
            batch_size=batch_size,
        )
        mpc_time = time.time() - mpc_start
        
        # Party 1 不应该得到实际结果
        dummy = torch.tensor(0.0)
        return dummy, mpc_time

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
        return x, x1, 0.0


class SecureTr(nn.Module):
    """带 MPC 的安全 Tr 类"""
    def __init__(self, hid_dim, dropout, party_id, use_mpc=True, batch_size=100):
        super(SecureTr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim)
        self.party_id = party_id
        self.use_mpc = use_mpc
        self.batch_size = batch_size
        
        if use_mpc:
            self.mpc_manager = MPSPDZManager(party_id=party_id)
        else:
            self.mpc_manager = None
    
    def forward(self, x, compute_norm=True):
        z = x
        x = self.fc1(x)
        y = x
        
        if not compute_norm:
            return x, torch.tensor(0.0), 0.0  # 修改：返回3个值
        
        if self.use_mpc and self.training and self.mpc_manager is not None:
            weights = self.fc1.weight.t()
            x1, mpc_time = get_x1_secure(z, None, weights, self.mpc_manager, batch_size=self.batch_size)
            return x, x1, mpc_time  # 返回3个值
        else:
            x1 = get_x1(z, y)
            return x, x1, 0.0  # 修改：返回3个值（mpc_time=0.0）


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
    
    def forward(self, x, compute_norm=True):
        m = x
        x, _, _ = self.tran[0](m[0])  # 修改：接收3个值
        x = self.leaky_relu[0](x)
        
        for i in range(1, self.r):
            y, _, _ = self.tran[i](m[i])  # 修改：接收3个值
            x = torch.cat([x, self.leaky_relu[i](y)], dim=1)
        
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits, 0.0  # 添加mpc_time=0.0


class SecureGMLP(nn.Module):
    """带 MPC 和 HE 的安全 GMLP 类"""
    def __init__(self, r, nfeat, nhid, nclass, dropout, drate, 
                 party_id, use_mpc=True, use_he=False, batch_size=100):
        super(SecureGMLP, self).__init__()
        self.nfeat = nfeat
        self.sum_l = sum(nfeat)
        self.nhid = nhid
        self.r = r
        self.party_id = party_id
        self.use_mpc = use_mpc
        self.use_he = use_he
        self.batch_size = batch_size
        
        # 使用 SecureTr 替代 Tr
        if use_mpc:
            self.tran = nn.ModuleList([
                SecureTr(self.nfeat[i], dropout, party_id, 
                        use_mpc=True, batch_size=batch_size).cuda() 
                for i in range(self.r)
            ])
            print(f"[SecureGMLP] Party {party_id} using SecureTr WITH MPC")
        else:
            # ⭐ 纯HE模式:使用普通Tr(只做本地计算)
            self.tran = nn.ModuleList([
                Tr(self.nfeat[i], dropout).cuda() 
                for i in range(self.r)
            ])
            print(f"[SecureGMLP] Party {party_id} using Tr WITHOUT MPC")
        
        self.classifier = Mlp(self.sum_l, self.nhid, nclass, dropout)
        self.leaky_relu = nn.ModuleList([
            nn.LeakyReLU(drate) 
            for i in range(self.r)
        ])
        
        # 初始化HE聚合器（如果启用）
        if use_he:
            self.grad_aggregator = GradientAggregator(party_id, use_he=True)
            print(f"[SecureGMLP] Party {party_id} initialized WITH HE")
        else:
            self.grad_aggregator = None
            print(f"[SecureGMLP] Party {party_id} initialized WITHOUT HE")
    
    def forward(self, x, compute_norm=True):
        m = x
        total_mpc_time = 0.0  # 添加这行
        # ⭐ 修改:根据use_mpc决定是否计算norm
        if self.use_mpc:
            x, _, mpc_time = self.tran[0](m[0], compute_norm=compute_norm)  # 接收3个值
            total_mpc_time += mpc_time
            x = self.leaky_relu[0](x)
        
            for i in range(1, self.r):
                y, _, mpc_time = self.tran[i](m[i], compute_norm=compute_norm)  # 接收3个值
                total_mpc_time += mpc_time
                x = torch.cat([x, self.leaky_relu[i](y)], dim=1)

        else:
            # 纯HE模式:使用普通Tr的forward
            # ⭐ 修改：接收3个值（因为Tr现在也返回3个值）
            x, _, mpc_time = self.tran[0](m[0])  # 修改这里
            x = self.leaky_relu[0](x)
            
            for i in range(1, self.r):
                y, _, mpc_time = self.tran[i](m[i])  # 修改这里
                x = torch.cat([x, self.leaky_relu[i](y)], dim=1)
        
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits, total_mpc_time  # 返回结果和总MPC时间
    
    def aggregate_gradients_he(self, local_gradients):
        """
        使用HE聚合梯度（仅在use_he=True时）
        
        Args:
            local_gradients: list of dict, 每个dict是一个客户端的梯度
        
        Returns:
            aggregated_grads: dict, 聚合后的梯度
        """
        if not self.use_he or self.grad_aggregator is None:
            print(f"[SecureGMLP] HE not enabled, skipping HE aggregation")
            return None, 0.0  #  确保返回2个值
        
        #  grad_aggregator.aggregate_gradients已经返回2个值
        return self.grad_aggregator.aggregate_gradients(local_gradients)