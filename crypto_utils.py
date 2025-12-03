import torch
import tenseal as ts
import numpy as np

class BFVEncryption:
    """BFV全同态加密方案 - 对应论文3.3节"""
    def __init__(self):
        # 生成BFV上下文
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=8192,
            plain_modulus=1032193
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
    def encrypt_tensor(self, tensor):
        """加密张量 - 对应论文中的Enc_pk操作"""
        # 转换为整数(乘以1000保留3位小数,对应论文中的×1000)
        tensor_int = (tensor.cpu().numpy() * 1000).astype(np.int64)
        flat = tensor_int.flatten().tolist()
        encrypted = ts.bfv_vector(self.context, flat)
        return encrypted, tensor.shape
    
    def decrypt_tensor(self, encrypted, shape):
        """解密张量 - 对应论文中的Dec_sk操作"""
        decrypted = encrypted.decrypt()
        tensor = torch.tensor(decrypted).reshape(shape) / 1000.0
        return tensor
    
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
        
        Args:
            x_shape: 输入特征的形状 (batch_size, feature_dim)
            w_shape: 权重矩阵的形状 (feature_dim, output_dim)
        
        Returns:
            list of triplets for each client
        """
        batch_size, feature_dim = x_shape
        _, output_dim = w_shape
        
        # 生成随机矩阵 a 和 b
        a = torch.randn(batch_size, feature_dim)  # 与 x 同形状
        b = torch.randn(feature_dim, output_dim)  # 与 w 同形状
        c = a @ b  # 结果是 (batch_size, output_dim)
        
        # 秘密分享
        a_shares = self._secret_share(a)
        b_shares = self._secret_share(b)
        c_shares = self._secret_share(c)
        
        return list(zip(a_shares, b_shares, c_shares))
    
    def _secret_share(self, tensor):
        """加法秘密分享 - 对应论文中的[[·]]符号"""
        shares = []
        remaining = tensor.clone()
        
        for i in range(self.num_clients - 1):
            share = torch.randn_like(tensor)
            shares.append(share)
            remaining -= share
        
        shares.append(remaining)
        return shares
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