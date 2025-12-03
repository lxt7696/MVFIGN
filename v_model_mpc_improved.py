# 改进的模型文件，集成 MPC 功能
# 文件路径: /home/lxt/project/MVFIGN/v_model_mpc_improved.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch.autograd import Variable
from mpc_utils import MPSPDZManager  # 使用修复后的工具类


class Mlp(nn.Module):
    """多层感知机模块"""
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


def get_x1_secure(party_features, weights, mpc_manager, session_id=0):
    """
    使用 MPC 安全计算范数
    
    改进版本 - 只需要本方的特征和权重
    
    Args:
        party_features: 本方特征 (m, n)
        weights: 权重矩阵 (n, n)
        mpc_manager: MPC 管理器
        session_id: 会话 ID（用于多次调用）
    
    Returns:
        安全计算的范数值
    """
    party_id = mpc_manager.party_id
    
    try:
        # 准备数据
        if party_id == 0:
            # Party 0 提供 A (party_features) 和 W (weights)
            result = mpc_manager.secure_matrix_multiply(
                A=party_features,
                W=weights,
                session_id=session_id
            )
        else:
            # Party 1 提供 B (party_features)
            result = mpc_manager.secure_matrix_multiply(
                B=party_features,
                session_id=session_id
            )
        
        # 计算范数
        x1 = torch.norm(result, p='fro')
        return x1
        
    except Exception as e:
        print(f"[WARNING] MPC computation failed: {e}")
        print("[WARNING] Falling back to local computation (INSECURE)")
        # 降级到本地计算（不安全，仅用于调试）
        z = party_features
        y = torch.matmul(party_features, weights.t())
        return get_x1(z, y)


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
    def __init__(self, hid_dim, dropout, party_id, use_mpc=True):
        super(SecureTr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim)
        self.party_id = party_id
        self.use_mpc = use_mpc
        self.session_counter = 0  # 用于跟踪 MPC 调用次数
        
        # 初始化 MPC 管理器
        if use_mpc:
            self.mpc_manager = MPSPDZManager(party_id=party_id)
            # 编译一次 MPC 程序
            try:
                self.mpc_manager.compile_program("secure_matmul_fixed")
                print(f"[SecureTr] Party {party_id} MPC program compiled")
            except Exception as e:
                print(f"[SecureTr] WARNING: MPC compilation failed: {e}")
                self.use_mpc = False
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
        
        # 是否使用 MPC
        if self.use_mpc and self.training:
            try:
                weights = self.fc1.weight.t()  # (hid_dim, hid_dim)
                x1 = get_x1_secure(
                    party_features=z,
                    weights=weights,
                    mpc_manager=self.mpc_manager,
                    session_id=self.session_counter
                )
                self.session_counter += 1
            except Exception as e:
                print(f"[SecureTr] MPC failed: {e}, using local computation")
                x1 = get_x1(z, y)
        else:
            # 本地计算
            x1 = get_x1(z, y)
        
        return x, x1
    
    def cleanup_mpc(self):
        """清理 MPC 临时文件"""
        if self.use_mpc and self.mpc_manager is not None:
            for i in range(self.session_counter):
                self.mpc_manager.cleanup(session_id=i)


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
                 party_id, use_mpc=True):
        super(SecureGMLP, self).__init__()
        self.nfeat = nfeat
        self.sum_l = sum(nfeat)
        self.nhid = nhid
        self.r = r
        self.party_id = party_id
        self.use_mpc = use_mpc
        
        # 使用 SecureTr 替代 Tr
        if use_mpc:
            print(f"[SecureGMLP] Initializing with MPC for Party {party_id}")
            self.tran = nn.ModuleList([
                SecureTr(self.nfeat[i], dropout, party_id, use_mpc=True).cuda() 
                for i in range(self.r)
            ])
        else:
            print(f"[SecureGMLP] Initializing WITHOUT MPC")
            self.tran = nn.ModuleList([
                Tr(self.nfeat[i], dropout).cuda() 
                for i in range(self.r)
            ])
        
        self.classifier = Mlp(self.sum_l, self.nhid, nclass, dropout)
        self.leaky_relu = nn.ModuleList([
            nn.LeakyReLU(drate) 
            for i in range(self.r)
        ])
        
        print(f"[SecureGMLP] Initialized successfully")
    
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
    
    def cleanup_mpc(self):
        """清理所有 MPC 临时文件"""
        if self.use_mpc:
            for tran_module in self.tran:
                if isinstance(tran_module, SecureTr):
                    tran_module.cleanup_mpc()


# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Secure Model Modules")
    print("=" * 60)
    
    # 测试参数
    party_id = 0
    nfeat = [128, 128]
    nhid = 256
    nclass = 7
    dropout = 0.6
    drate = 1.0
    
    # 创建模型
    print("\nCreating SecureGMLP model...")
    model = SecureGMLP(
        r=2, 
        nfeat=nfeat, 
        nhid=nhid, 
        nclass=nclass,
        dropout=dropout, 
        drate=drate,
        party_id=party_id,
        use_mpc=False  # 设置为 True 启用 MPC
    ).cuda()
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    batch_size = 32
    test_features = [
        torch.randn(batch_size, nfeat[0]).cuda(),
        torch.randn(batch_size, nfeat[1]).cuda()
    ]
    
    output = model(test_features, compute_norm=False)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {nclass})")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)