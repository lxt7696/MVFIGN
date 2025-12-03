# crypto_utils.py - 完整版（新增梯度聚合加密）

import torch
import tenseal as ts
import numpy as np

class BFVEncryption:
    """BFV全同态加密方案 - 对应论文3.3节"""
    def __init__(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=8192,
            plain_modulus=1032193
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
    def encrypt_tensor(self, tensor):
        """加密张量"""
        original_shape = tensor.shape
        # 转换为整数
        tensor_int = (tensor.cpu().numpy() * 1000).astype(np.int64)
        flat = tensor_int.flatten().tolist()
        
        # 分批加密（BFV有长度限制）
        batch_size = 4096
        encrypted_batches = []
        for i in range(0, len(flat), batch_size):
            batch = flat[i:i+batch_size]
            enc_batch = ts.bfv_vector(self.context, batch)
            encrypted_batches.append(enc_batch)
        
        return encrypted_batches, original_shape
    
    def decrypt_tensor(self, encrypted_data, shape):
        """解密张量 - 修复版"""
        # encrypted_data 可能是单个密文或密文列表
        if isinstance(encrypted_data, list):
            # 如果是列表（分批加密的情况）
            decrypted_parts = []
            for enc_batch in encrypted_data:
                decrypted_parts.extend(enc_batch.decrypt())
            tensor = torch.tensor(decrypted_parts).reshape(shape) / 1000.0
        else:
            # 如果是单个密文（小张量的情况）
            decrypted = encrypted_data.decrypt()
            tensor = torch.tensor(decrypted).reshape(shape) / 1000.0
        
        return tensor
    
    def aggregate_encrypted_gradients(self, encrypted_grads):
        """
        聚合加密梯度 - 对应论文Algorithm 3
        
        Args:
            encrypted_grads: list of (encrypted_grad, shape) tuples
        Returns:
            aggregated encrypted gradient
        """
        if not encrypted_grads:
            return None, None
            
        # 利用同态加性：Enc(g1) + Enc(g2) = Enc(g1 + g2)
        result_enc, result_shape = encrypted_grads[0]
        
        for enc_grad, _ in encrypted_grads[1:]:
            result_enc = result_enc + enc_grad
        
        return result_enc, result_shape
    
    def get_public_context(self):
        """获取公钥上下文"""
        return self.context.serialize()


class MPCBeaver:
    """Beaver三元组MPC方案 - 对应论文3.2节"""
    def __init__(self, num_clients):
        self.num_clients = num_clients
        
    def generate_triplet(self, x_shape, w_shape):
        """
        生成Beaver三元组 [[a]], [[b]], [[c]] where c = a @ b
        
        注意：每次批量训练时应重新生成以匹配批量大小
        """
        batch_size, feature_dim = x_shape
        _, output_dim = w_shape
        
        # 生成随机矩阵 a 和 b
        a = torch.randn(batch_size, feature_dim)
        b = torch.randn(feature_dim, output_dim)
        c = a @ b
        
        # 秘密分享
        a_shares = self._secret_share(a)
        b_shares = self._secret_share(b)
        c_shares = self._secret_share(c)
        
        return list(zip(a_shares, b_shares, c_shares))
    
    def _secret_share(self, tensor):
        """加法秘密分享"""
        shares = []
        remaining = tensor.clone()
        
        for i in range(self.num_clients - 1):
            share = torch.randn_like(tensor)
            shares.append(share)
            remaining -= share
        
        shares.append(remaining)
        return shares

    def reveal(self, shares):
        """重构秘密 - 对应论文中的reveal步骤"""
        return sum(shares)
    
    def secure_matmul(self, x_shares, w_shares, triplet_shares, client_id):
        """
        使用Beaver三元组进行安全矩阵乘法（修复版）
        
        Args:
            x_shares: 所有客户端的x份额列表 [x_share_0, x_share_1, ...]
            w_shares: 所有客户端的w份额列表 [w_share_0, w_share_1, ...]
            triplet_shares: 当前客户端的三元组份额 (a_share, b_share, c_share)
            client_id: 当前客户端ID
        
        Returns:
            当前客户端的结果份额
        """
        a_share, b_share, c_share = triplet_shares
        x_share = x_shares[client_id]
        w_share = w_shares[client_id]
        
        # 确保设备一致
        device = x_share.device
        a_share = a_share.to(device)
        b_share = b_share.to(device)
        c_share = c_share.to(device)
        
        # 步骤1: 计算 α = x - a（需要reveal）
        alpha_share = x_share - a_share
        # 在实际系统中，这里需要通信reveal alpha
        # 为了模拟，假设我们可以访问所有份额
        alpha = self.reveal([x_shares[i] - triplet_shares[i][0].to(device) 
                            for i in range(self.num_clients)])
        
        # 步骤2: 计算 β = w - b（需要reveal）
        beta_share = w_share - b_share
        beta = self.reveal([w_shares[i] - triplet_shares[i][1].to(device) 
                           for i in range(self.num_clients)])
        
        # 步骤3: 计算结果份额（对应论文公式4）
        z_share = c_share + alpha @ b_share + a_share @ beta
        
        # 每个客户端只加一次 alpha @ beta / num_clients
        if client_id == 0:
            z_share = z_share + (alpha @ beta)
        
        return z_share