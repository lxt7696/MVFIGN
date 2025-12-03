import torch
import numpy as np
import time

try:
    import tenseal as ts
    from tenseal import Context, SCHEME_TYPE, CKKSVector
    HE_AVAILABLE = True
except ImportError:
    print("[WARNING] TenSEAL not installed. HE functionality disabled.")
    HE_AVAILABLE = False


class HEManager:
    """同态加密管理器 - 用于安全梯度聚合"""
    
    def __init__(self, party_id):
        if not HE_AVAILABLE:
            raise RuntimeError("TenSEAL not available. Install with: pip install tenseal")
        self.party_id = party_id
        self.context = None
        self.setup_context()
        print(f"[HEManager] Party {party_id} initialized with HE support")
    
    def setup_context(self):
        """设置CKKS上下文"""
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        print(f"[HEManager] CKKS context created with poly_modulus_degree=8192")
    
    def encrypt_gradient(self, grad_tensor):
        """加密梯度张量"""
        grad_flat = grad_tensor.detach().cpu().numpy().flatten()
        encrypted = ts.ckks_vector(self.context, grad_flat)
        print(f"[HEManager - Party {self.party_id}] Encrypted gradient: shape={grad_tensor.shape}, size={len(grad_flat)}")
        return encrypted
    
    def decrypt_gradient(self, encrypted_grad, original_shape):
        """解密梯度"""
        decrypted_flat = encrypted_grad.decrypt()
        grad_tensor = torch.tensor(decrypted_flat, dtype=torch.float32).reshape(original_shape)
        print(f"[HEManager - Party {self.party_id}] Decrypted gradient: shape={original_shape}")
        return grad_tensor
    
    def aggregate_encrypted_gradients(self, encrypted_grads):
        """聚合加密梯度 (同态加法)"""
        if len(encrypted_grads) == 0:
            return None
        
        result = encrypted_grads[0]
        for enc_grad in encrypted_grads[1:]:
            result += enc_grad
        
        print(f"[HEManager - Party {self.party_id}] Aggregated {len(encrypted_grads)} encrypted gradients")
        return result
    
    def serialize_context(self):
        """序列化上下文用于传输"""
        return self.context.serialize()
    
    def load_context(self, serialized_context):
        """加载序列化的上下文"""
        self.context = ts.context_from(serialized_context)
        print(f"[HEManager - Party {self.party_id}] Loaded serialized context")

class GradientAggregator:
    """使用HE的梯度聚合器"""
    
    def __init__(self, party_id, use_he=True):
        self.party_id = party_id
        self.use_he = use_he
        
        if use_he:
            self.he_manager = HEManager(party_id)
            print(f"[GradientAggregator] Party {party_id} initialized WITH HE")
        else:
            self.he_manager = None
            print(f"[GradientAggregator] Party {party_id} initialized WITHOUT HE")
    
    def aggregate_gradients(self, local_gradients):
        """
        聚合多个本地梯度
        
        Args:
            local_gradients: list of dict, 每个dict包含模型参数的梯度
        
        Returns:
            aggregated_grads: dict, 聚合后的梯度
            he_time: dict or float, HE操作时间详情（或总时间）
        """
        if not self.use_he:
            # 明文聚合
            return self._plaintext_aggregate(local_gradients), 0.0
        
        # HE聚合
        return self._he_aggregate(local_gradients)
    
    def _plaintext_aggregate(self, local_gradients):
        """明文梯度聚合 (平均)"""
        if len(local_gradients) == 0:
            return {}, 0.0
        
        aggregated = {}
        num_clients = len(local_gradients)
        
        for key in local_gradients[0].keys():
            grad_sum = torch.zeros_like(local_gradients[0][key])
            for grads in local_gradients:
                if grads[key] is not None:
                    grad_sum += grads[key]
            aggregated[key] = grad_sum / num_clients
        
        print(f"[GradientAggregator - Party {self.party_id}] Plaintext aggregation: {len(aggregated)} parameters")
        return aggregated
    
    def _he_aggregate(self, local_gradients):
        """HE梯度聚合"""
        if len(local_gradients) == 0:
            return {}, {'total': 0.0, 'encrypt': 0.0, 'aggregate': 0.0, 'decrypt': 0.0}
        
        print(f"[GradientAggregator - Party {self.party_id}] Starting HE aggregation for {len(local_gradients)} clients")
        
        he_start = time.time()
        encrypt_time = 0.0
        aggregate_time = 0.0
        decrypt_time = 0.0

        aggregated = {}
        num_clients = len(local_gradients)
        
        # 收集所有客户端中存在的参数名
        all_keys = set()
        for grads in local_gradients:
            all_keys.update(grads.keys())
        
        for key in all_keys:
            # 收集所有客户端的该参数梯度
            grads_to_aggregate = []
            shapes = []
            
            for grads in local_gradients:
                if key in grads and grads[key] is not None:
                    grads_to_aggregate.append(grads[key])
                    shapes.append(grads[key].shape)
            
            if len(grads_to_aggregate) == 0:
                aggregated[key] = None
                continue
            
            # ⭐ 修改：加密每个梯度 - 添加计时
            encrypt_start = time.time()
            encrypted_grads = []
            for grad in grads_to_aggregate:
                enc_grad = self.he_manager.encrypt_gradient(grad)
                encrypted_grads.append(enc_grad)
            encrypt_time += time.time() - encrypt_start
            
            # ⭐ 修改：同态聚合 - 添加计时
            aggregate_start = time.time()
            aggregated_encrypted = self.he_manager.aggregate_encrypted_gradients(encrypted_grads)
            aggregate_time += time.time() - aggregate_start
            
            # ⭐ 修改：解密结果 - 添加计时
            decrypt_start = time.time()
            aggregated_grad = self.he_manager.decrypt_gradient(aggregated_encrypted, shapes[0])
            decrypt_time += time.time() - decrypt_start
            
            # 平均（使用实际参与聚合的客户端数量）
            aggregated[key] = aggregated_grad / len(grads_to_aggregate)
            
            # 转回CUDA
            if torch.cuda.is_available():
                aggregated[key] = aggregated[key].cuda()
        
        he_time = time.time() - he_start
        
        print(f"[GradientAggregator - Party {self.party_id}] HE aggregation complete: {len(aggregated)} parameters")
        print(f"[GradientAggregator - Party {self.party_id}] HE Time Breakdown:")
        print(f"  - Encryption: {encrypt_time:.4f}s")
        print(f"  - Aggregation: {aggregate_time:.4f}s")
        print(f"  - Decryption: {decrypt_time:.4f}s")
        print(f"  - Total HE: {he_time:.4f}s")
        
        # ⭐ 修改：返回详细的时间统计（而不是总时间）
        return aggregated, {
            'total': he_time,
            'encrypt': encrypt_time,
            'aggregate': aggregate_time,
            'decrypt': decrypt_time
        }