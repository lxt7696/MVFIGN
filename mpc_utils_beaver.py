import torch
import subprocess
import numpy as np
import os
import socket
import struct

class MPSPDZManager:
    """
    MP-SPDZ管理器 - 与MP-SPDZ进程通信
    实现论文Algorithm 2中的安全矩阵乘法
    """
    
    def __init__(self, party_id, num_parties=2, 
                 mpspdz_path="/home/lxt/project/MVFIGN/MP-SPDZ",
                 protocol="semi"):
        """
        初始化MPC管理器
        
        Args:
            party_id: 当前参与方ID (0或1)
            num_parties: 总参与方数量
            mpspdz_path: MP-SPDZ安装路径
            protocol: 使用的协议 (semi/mascot/semi2k)
        """
        self.party_id = party_id
        self.num_parties = num_parties
        self.mpspdz_path = mpspdz_path
        self.protocol = protocol
        
        # 输入输出目录
        self.player_data_dir = f"{mpspdz_path}/Player-Data"
        os.makedirs(self.player_data_dir, exist_ok=True)
        
        print(f"[MPSPDZManager] Party {party_id} initialized with {protocol} protocol")
    
    def _secret_share(self, tensor):
        """
        对张量进行加法秘密分享
        
        Args:
            tensor: 要分享的张量
            
        Returns:
            shares: 份额列表
        """
        shares = []
        remaining = tensor.clone()
        
        for i in range(self.num_parties - 1):
            share = torch.randn_like(tensor) * 0.1  # 小的随机噪声
            shares.append(share)
            remaining = remaining - share
        
        shares.append(remaining)
        return shares
    
    def _tensor_to_fixed_point(self, tensor, precision=16):
        """
        将浮点张量转换为定点数
        
        Args:
            tensor: 输入张量
            precision: 小数位数
            
        Returns:
            整数数组
        """
        scale = 2 ** precision
        fixed = (tensor * scale).long()
        return fixed.cpu().numpy()
    
    def _fixed_point_to_tensor(self, fixed_array, shape, precision=16):
        """
        将定点数转换回浮点张量
        
        Args:
            fixed_array: 整数数组
            shape: 目标形状
            precision: 小数位数
            
        Returns:
            浮点张量
        """
        scale = 2 ** precision
        tensor = torch.tensor(fixed_array, dtype=torch.float32) / scale
        return tensor.reshape(shape)
    
    def _write_input_to_file(self, party_id, channel, data, precision=16):
        """
        写入输入数据到MP-SPDZ格式文件
        
        Args:
            party_id: 参与方ID
            channel: 输入通道编号
            data: numpy数组或torch张量
            precision: 定点数精度
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        filename = f"{self.player_data_dir}/Input-P{party_id}-{channel}"
        
        # 转换为定点数
        scale = 2 ** precision
        fixed_data = (data.flatten() * scale).astype(np.int64)
        
        # MP-SPDZ的二进制输入格式
        with open(filename, 'wb') as f:
            for val in fixed_data:
                # 写入64位整数（小端序）
                f.write(struct.pack('<q', val))
        
        print(f"[Party {party_id}] Wrote {len(fixed_data)} values to {filename}")
    
    def _read_output_from_file(self, party_id, channel, shape, precision=16):
        """
        从MP-SPDZ输出文件读取结果
        
        Args:
            party_id: 参与方ID
            channel: 输出通道编号
            shape: 期望的形状
            precision: 定点数精度
            
        Returns:
            torch张量
        """
        filename = f"{self.player_data_dir}/Output-P{party_id}-{channel}"
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Output file not found: {filename}")
        
        # 读取二进制输出
        values = []
        with open(filename, 'rb') as f:
            while True:
                data = f.read(8)  # 读取8字节（64位整数）
                if not data:
                    break
                val = struct.unpack('<q', data)[0]
                values.append(val)
        
        # 转换回浮点数
        scale = 2 ** precision
        result = torch.tensor(values, dtype=torch.float32) / scale
        
        if shape is not None:
            result = result.reshape(shape)
        
        print(f"[Party {party_id}] Read {len(values)} values from {filename}")
        return result
    
    def secure_matrix_multiply(self, A=None, B=None, W=None):
        """
        使用MP-SPDZ进行安全矩阵乘法: (A - B) * W
        
        Args:
            A: Party 0的矩阵 (m, n)，仅Party 0提供
            B: Party 1的矩阵 (m, n)，仅Party 1提供  
            W: 权重矩阵 (n, k)，Party 0提供
            
        Returns:
            result: 计算结果 (m, k)
        """
        
        # 清理旧的输入输出文件
        for f in os.listdir(self.player_data_dir):
            if f.startswith('Input-') or f.startswith('Output-'):
                os.remove(os.path.join(self.player_data_dir, f))
        
        # 准备输入
        output_shape = None
        
        if self.party_id == 0:
            if A is None or W is None:
                raise ValueError("Party 0 must provide A and W")
            
            # Party 0对A和W进行秘密分享
            A_shares = self._secret_share(A)
            W_shares = self._secret_share(W)
            
            # 写入Party 0的份额
            self._write_input_to_file(0, 0, A_shares[0])
            self._write_input_to_file(0, 1, W_shares[0])
            
            output_shape = (A.shape[0], W.shape[1])
            
            print(f"[Party 0] Input shape A: {A.shape}, W: {W.shape}")
            
            # TODO: 在实际部署中，需要将份额发送给Party 1
            # 这里简化：假设通过共享文件系统
            
        elif self.party_id == 1:
            if B is None:
                raise ValueError("Party 1 must provide B")
            
            # Party 1对B进行秘密分享
            B_shares = self._secret_share(B)
            
            # 写入Party 1的份额
            self._write_input_to_file(1, 0, B_shares[1])
            
            print(f"[Party 1] Input shape B: {B.shape}")
            
            # TODO: 接收来自Party 0的份额
        
        # 运行MP-SPDZ程序
        cmd = [
            f"Scripts/{self.protocol}-party.x",
            "-p", str(self.party_id),
            "-N", str(self.num_parties),
            "secure_matmul_beaver"
        ]
        
        print(f"[Party {self.party_id}] Starting MPC computation...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.mpspdz_path,
                capture_output=True, 
                text=True,
                timeout=60  # 60秒超时
            )
            
            if result.returncode != 0:
                print(f"[Party {self.party_id}] STDERR:\n{result.stderr}")
                print(f"[Party {self.party_id}] STDOUT:\n{result.stdout}")
                raise RuntimeError(f"MP-SPDZ execution failed with code {result.returncode}")
            
            print(f"[Party {self.party_id}] MPC computation completed")
            print(f"Output:\n{result.stdout}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("MP-SPDZ execution timed out")
        
        # 读取输出（如果有）
        try:
            output_file = f"{self.player_data_dir}/Output-P{self.party_id}-0"
            if os.path.exists(output_file):
                result_tensor = self._read_output_from_file(
                    self.party_id, 0, output_shape
                )
                return result_tensor
            else:
                print(f"[Party {self.party_id}] No output file generated")
                return None
                
        except Exception as e:
            print(f"[Party {self.party_id}] Error reading output: {e}")
            return None


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mpc_utils.py <party_id>")
        print("Example: python mpc_utils.py 0")
        sys.exit(1)
    
    party_id = int(sys.argv[1])
    
    print(f"\n{'='*50}")
    print(f"Testing MP-SPDZ Manager - Party {party_id}")
    print(f"{'='*50}\n")
    
    # 创建测试数据
    m, n, k = 4, 4, 4
    
    manager = MPSPDZManager(
        party_id=party_id,
        mpspdz_path="/home/lxt/project/MVFIGN/MP-SPDZ"
    )
    
    if party_id == 0:
        A = torch.randn(m, n)
        W = torch.randn(n, k)
        
        print("Party 0 inputs:")
        print(f"A = {A}")
        print(f"W = {W}")
        
        result = manager.secure_matrix_multiply(A=A, W=W)
        
    else:  # party_id == 1
        B = torch.randn(m, n)
        
        print("Party 1 inputs:")
        print(f"B = {B}")
        
        result = manager.secure_matrix_multiply(B=B)
    
    if result is not None:
        print(f"\nParty {party_id} result:")
        print(result)
    else:
        print(f"\nParty {party_id}: No result returned")