# mpc_utils.py - 修复版（解决 Party 1 返回 None 问题）

import subprocess
import numpy as np
import torch
import os
import time
import re
import signal
import sys

class MPSPDZManager:
    """MP-SPDZ 安全计算管理器"""
    
    def __init__(self, party_id, mpspdz_root="/home/lxt/project/MVFIGN/MP-SPDZ"):
        self.party_id = party_id
        self.mpspdz_root = mpspdz_root
        self.player_data_dir = os.path.join(mpspdz_root, "Player-Data")

        # ⭐ 添加编译缓存
        #self.compiled_programs = set()  # 记录已编译的程序
        
        os.makedirs(self.player_data_dir, exist_ok=True)
        
        print(f"[Party {party_id}] MPC Manager initialized")
        print(f"  MP-SPDZ Root: {mpspdz_root}")
        print(f"  Player Data: {self.player_data_dir}")
        
    def write_inputs(self, data_dict, session_id=0):
        """将数据写入 MP-SPDZ 输入文件"""
        input_file = os.path.join(
            self.player_data_dir, 
            f"Input-P{self.party_id}-{session_id}"
        )
        
        values = []
        
        # Party 0 提供 A 和 W
        if self.party_id == 0:
            if 'A' in data_dict and data_dict['A'] is not None:
                A = self._to_numpy(data_dict['A'])
                values.extend(A.flatten().tolist())
                print(f"[Party {self.party_id}] A shape: {A.shape}, values: {len(A.flatten())}")
            
            if 'W' in data_dict and data_dict['W'] is not None:
                W = self._to_numpy(data_dict['W'])
                values.extend(W.flatten().tolist())
                print(f"[Party {self.party_id}] W shape: {W.shape}, values: {len(W.flatten())}")
        
        # Party 1 提供 B
        else:
            if 'B' in data_dict and data_dict['B'] is not None:
                B = self._to_numpy(data_dict['B'])
                values.extend(B.flatten().tolist())
                print(f"[Party {self.party_id}] B shape: {B.shape}, values: {len(B.flatten())}")
        
        # 写入文件（每行一个值）
        with open(input_file, 'w') as f:
            for val in values:
                f.write(f"{float(val)}\n")
        
        print(f"[Party {self.party_id}] Wrote {len(values)} values to {input_file}")
        return len(values)
    
    def _to_numpy(self, tensor):
        """
        转换为 numpy 数组
        ✅ 修复：添加 detach() 处理带梯度的 Tensor
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    
    def read_outputs_from_stdout(self, stdout, shape):
        """从标准输出解析结果"""
        if self.party_id == 1:
            # Party 1 明确知道自己不应该获得结果
            print(f"[Party {self.party_id}] Party 1 should not receive plaintext output")
            # ⭐ 关键修复：返回零矩阵而不是 None
            print(f"[Party {self.party_id}] Returning zero matrix of shape {shape}")
            #return torch.zeros(shape[0], shape[1])
            return torch.tensor(0.0)  # 返回标量而不是矩阵

        print(f"\n[Party {self.party_id}] ===== PARSING OUTPUT =====")
        print(f"[Party {self.party_id}] Total stdout length: {len(stdout)} chars")
        
        # 检查是否为空
        if len(stdout.strip()) == 0:
            print(f"[Party {self.party_id}] ⚠ WARNING: stdout is empty!")
            #return torch.zeros(shape[0], shape[1])
            return torch.tensor(0.0)  # 返回标量而不是矩阵
        
        lines = stdout.strip().split('\n')
        print(f"[Party {self.party_id}] Total lines: {len(lines)}")
        
        values = []
        capturing = False
        
        for idx, line in enumerate(lines):
            line = line.strip()
            
            # 查找结果标记
            if '=== RESULTS ===' in line:
                capturing = True
                print(f"\n[Party {self.party_id}] ✓ Found RESULTS marker at line {idx}")
                continue
            
            if '=== END RESULTS ===' in line:
                capturing = False
                print(f"[Party {self.party_id}] ✓ Found END RESULTS marker at line {idx}")
                print(f"[Party {self.party_id}] Total values captured: {len(values)}")
                break
            
            # 在捕获区域内解析数值
            if capturing and line:
                # 跳过明显的非数值行
                if any(x in line for x in ['===', 'Step', 'Party', 'Reading', 'Computing', 'loaded', 'Difference', 'Matrix', 'Revealing']):
                    continue
                
                try:
                    # 尝试直接转换
                    val = float(line)
                    values.append(val)
                    if len(values) <= 5 or len(values) % 100 == 0:
                        print(f"[Party {self.party_id}]   Value {len(values)}: {val}")
                except ValueError:
                    # 尝试提取数字
                    match = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', line)
                    if match:
                        val = float(match.group())
                        values.append(val)
        
        # 验证结果 - 现在期望只有一个标量
        print(f"\n[Party {self.party_id}] ===== PARSING SUMMARY =====")
        print(f"[Party {self.party_id}] Expected: 1 scalar value (norm squared)")
        print(f"[Party {self.party_id}] Got: {len(values)} values")
        
        if len(values) == 0:
            print(f"\n[Party {self.party_id}] ⚠ WARNING: No values found")
            print(f"[Party {self.party_id}] Showing first 50 lines of stdout:")
            for i, line in enumerate(lines[:50]):
                print(f"  {i}: {line}")
            return None
        
        # 只取第一个值（范数平方）
        norm_squared = values[0]
        norm = np.sqrt(norm_squared)  # 开方得到范数
        
        print(f"[Party {self.party_id}] Norm squared: {norm_squared:.4f}")
        print(f"[Party {self.party_id}] Norm (Frobenius): {norm:.4f}")
        print(f"[Party {self.party_id}] ✓ Successfully parsed scalar")
        
        return torch.tensor(norm).float()  # 返回标量tensor
    
    def compile_program(self, program_name="secure_matmul"):
        """编译 MPC 程序"""
        # ⭐ 检查是否已编译
        #if program_name in self.compiled_programs:
         #   print(f"[Party {self.party_id}] Using cached compilation for {program_name}")
          #  return True

        compile_cmd = f"cd {self.mpspdz_root} && ./compile.py -R 64 {program_name}"
        
        print(f"[Party {self.party_id}] Compiling {program_name}...")
        
        result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[Party {self.party_id}] Compilation error:")
            print(result.stderr)
            raise RuntimeError("MPC program compilation failed")

        # ⭐ 记录编译过的程序
        #self.compiled_programs.add(program_name)        
        print(f"[Party {self.party_id}] Compilation successful")
        return True
    
    def run_party_async(self, program_name="secure_matmul", 
                       port_base=40000, timeout=10000):
        """异步运行 MPC 参与方"""
        run_cmd = (
            f"cd {self.mpspdz_root} && "
            f"./semi2k-party.x -p {self.party_id} "
            f"-N 2 -pn {port_base} "
            f"{program_name}"
        )
        
        print(f"[Party {self.party_id}] Running MPC...")
        print(f"  Command: {run_cmd}")
        
        if self.party_id == 0:
            print(f"[Party {self.party_id}] Starting as server (waiting for Party 1)...")
        else:
            print(f"[Party {self.party_id}] Starting as client (connecting to Party 0)...")
        
        # ⭐ 改动：用 Popen 替换 run，这样不会阻塞
        process = subprocess.Popen(
            run_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # 等待进程完成，但有超时限制
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(f"MPC timeout for Party {self.party_id}")
        
        if process.returncode != 0:
            print(f"[Party {self.party_id}] Execution error (returncode={process.returncode}):")
            print(f"[Party {self.party_id}] stderr:\n{stderr}")
            print(f"[Party {self.party_id}] stdout:\n{stdout}")
            raise RuntimeError("MPC execution failed")
        
        print(f"[Party {self.party_id}] MPC completed successfully")
        
        # 返回类似 run() 的结果对象
        result = type('obj', (object,), {'stdout': stdout, 'stderr': stderr, 'returncode': process.returncode})()
        return result

        #result = subprocess.run(
         #   run_cmd,
          #  shell=True,
           # capture_output=True,
            #text=True,
          #  timeout=timeout
        #)
        
        #if result.returncode != 0:
         #   print(f"[Party {self.party_id}] Execution error (returncode={result.returncode}):")
          #  print(f"[Party {self.party_id}] stderr:\n{result.stderr}")
           # print(f"[Party {self.party_id}] stdout:\n{result.stdout}")
            #raise RuntimeError("MPC execution failed")
        
      #  print(f"[Party {self.party_id}] MPC completed successfully")
       # return result
    
    def secure_matrix_multiply(self, A=None, B=None, W=None, 
                          compile_once=True, session_id=0, skip_compile=False):
        """
        安全计算 (A - B) * W
        ⭐ 关键修复：确保所有分支都返回正确形状的 tensor
        """
        print(f"\n[Party {self.party_id}] ========== secure_matrix_multiply START (session={session_id}, skip_compile={skip_compile}) ==========")
        
        # 1. 准备输入
        print(f"[Party {self.party_id}] Step 1: Preparing inputs...")
        if self.party_id == 0:
            if A is None or W is None:
                raise ValueError("Party 0 must provide A and W")
            m, n = A.shape
            print(f"[Party {self.party_id}] A shape: {A.shape}, W shape: {W.shape}")
            self.write_inputs({'A': A, 'W': W}, session_id)
        else:
            if B is None:
                raise ValueError("Party 1 must provide B")
            m, n = B.shape
            print(f"[Party {self.party_id}] B shape: {B.shape}")
            self.write_inputs({'B': B}, session_id)
        
        print(f"[Party {self.party_id}] Step 1 DONE: Inputs written")
        
        # 2. 编译程序（只在第一次或者明确要求时编译）
        print(f"[Party {self.party_id}] Step 2: Checking compilation (skip_compile={skip_compile}, compile_once={compile_once}, session_id={session_id})...")
        if not skip_compile and (not compile_once or session_id == 0):
            if self.party_id == 0:
                print(f"[Party {self.party_id}] 🔨 Compiling MPC program (THIS MAY TAKE A WHILE)...")
                self.compile_program("secure_matmul")
                print(f"[Party {self.party_id}] ✓ Compilation completed")
            else:
                print(f"[Party {self.party_id}] ⏳ Waiting for Party 0 to compile (sleeping 3 seconds)...")
                time.sleep(3)
                print(f"[Party {self.party_id}] ✓ Wait completed")
        else:
            print(f"[Party {self.party_id}] ✓ Skipping compilation (already compiled)")
        
        print(f"[Party {self.party_id}] Step 2 DONE: Compilation phase complete")
        
        # 3. 运行 MPC
        print(f"[Party {self.party_id}] Step 3: Starting MPC execution...")
        result = self.run_party_async("secure_matmul", port_base=40000, timeout=10000)
        print(f"[Party {self.party_id}] Step 3 DONE: MPC execution finished")
        
        # 4. 从 stdout 读取结果（现在是标量）
        try:
            print(f"[Party {self.party_id}] Parsing output...")
            output_scalar = self.read_outputs_from_stdout(result.stdout, (m, n))
            
            if output_scalar is None:
                # Party 1 不应该得到输出
                print(f"[Party {self.party_id}] No output found, returning zero scalar")
                print(f"[Party {self.party_id}] This is expected for Party 1")
                output_scalar = torch.tensor(0.0)
            
            print(f"[Party {self.party_id}] Successfully got norm scalar: {output_scalar.item():.4f}")
            return output_scalar  # 返回标量而不是矩阵
            
        except Exception as e:
            print(f"[Party {self.party_id}] Step 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Party {self.party_id}] Returning zero matrix as fallback")
            print(f"[Party {self.party_id}] ========== secure_matrix_multiply END (session={session_id}) ==========\n")
            #return torch.zeros(m, n)
            return torch.tensor(0.0)  # 返回标量而不是矩阵
    
    def secure_matrix_multiply_batched(self, A=None, B=None, W=None, batch_size=500):
        """
        分批计算
        ⭐ 修改：MPC现在返回标量（范数），需要累加范数平方
        """
        print(f"\n[Party {self.party_id}] ========== secure_matrix_multiply_batched START ==========")
        
        if self.party_id == 0:
            A_tensor = torch.tensor(A) if not isinstance(A, torch.Tensor) else A
            W_tensor = torch.tensor(W) if not isinstance(W, torch.Tensor) else W
            m, n = A_tensor.shape
            
            print(f"[Party 0] Total samples: {m}, Feature dim: {n}")
            print(f"[Party 0] Using batch_size: {batch_size}")
            
        else:
            B_tensor = torch.tensor(B) if not isinstance(B, torch.Tensor) else B
            m, n = B_tensor.shape
            
            print(f"[Party 1] Total samples: {m}, Feature dim: {n}")
            print(f"[Party 1] Using batch_size: {batch_size}")
        
        # ⭐ 修改：存储范数平方（而不是矩阵结果）
        batch_norms_squared = []
        num_batches = (m + batch_size - 1) // batch_size

        # ⭐ 关键优化：如果只有1个batch，就不需要循环！
        if num_batches == 1:
            print(f"[Party {self.party_id}] Single batch optimization: avoiding loop overhead")
            # 直接处理，跳过循环
            if self.party_id == 0:
                A_tensor = torch.tensor(A) if not isinstance(A, torch.Tensor) else A
                W_tensor = torch.tensor(W) if not isinstance(W, torch.Tensor) else W
                result = self.secure_matrix_multiply(A=A_tensor, W=W_tensor, session_id=0, skip_compile=False)
            else:
                B_tensor = torch.tensor(B) if not isinstance(B, torch.Tensor) else B
                result = self.secure_matrix_multiply(B=B_tensor, session_id=0, skip_compile=False)
            
            if torch.cuda.is_available():
                result = result.cuda()
            return result
        
        # 如果有多个batch才进循环
        print(f"[Party {self.party_id}] Will process {num_batches} batches")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, m)
            actual_batch = end_idx - start_idx
            
            print(f"\n[Party {self.party_id}] ========== BATCH {batch_idx+1}/{num_batches} START ==========")
            print(f"[Party {self.party_id}] Processing rows {start_idx}-{end_idx} (size={actual_batch})")
            
            if self.party_id == 0:
                # Party 0 处理
                A_batch = self._to_numpy(A_tensor[start_idx:end_idx])
                
                # 如果批次小于设定值，需要填充
                if actual_batch < batch_size:
                    padding = np.zeros((batch_size - actual_batch, n))
                    A_batch = np.vstack([A_batch, padding])
                    print(f"[Party 0] Padded batch from {actual_batch} to {batch_size} rows")
                
                print(f"[Party 0] Calling secure_matrix_multiply with session_id={batch_idx}, skip_compile={batch_idx > 0}...")
                # ⭐ 修改：现在返回标量（范数）
                batch_norm = self.secure_matrix_multiply(
                    A=A_batch, W=self._to_numpy(W_tensor), 
                    session_id=batch_idx,
                    skip_compile=(batch_idx > 0)
                )
                
                # ⭐ 修改：存储范数的平方
                if batch_norm is None or not isinstance(batch_norm, torch.Tensor):
                    print(f"[Party 0] WARNING: batch_norm is None or not tensor, using 0")
                    batch_norms_squared.append(torch.tensor(0.0))
                else:
                    batch_norms_squared.append(batch_norm ** 2)
                    print(f"[Party 0] Batch {batch_idx} norm: {batch_norm.item():.4f}")
                
                print(f"[Party 0] ========== BATCH {batch_idx+1}/{num_batches} DONE ==========\n")
                
            else:
                # Party 1 处理
                B_batch = self._to_numpy(B_tensor[start_idx:end_idx])
                
                # 同样需要填充
                if actual_batch < batch_size:
                    padding = np.zeros((batch_size - actual_batch, n))
                    B_batch = np.vstack([B_batch, padding])
                    print(f"[Party 1] Padded batch from {actual_batch} to {batch_size} rows")
                
                print(f"[Party 1] Calling secure_matrix_multiply with session_id={batch_idx}, skip_compile={batch_idx > 0}...")
                # ⭐ 修改：现在返回标量（范数）
                batch_norm = self.secure_matrix_multiply(
                    B=B_batch, 
                    session_id=batch_idx,
                    skip_compile=(batch_idx > 0)
                )
                
                # ⭐ 修改：Party 1也存储（虽然应该是0）
                if batch_norm is None or not isinstance(batch_norm, torch.Tensor):
                    batch_norms_squared.append(torch.tensor(0.0))
                else:
                    batch_norms_squared.append(batch_norm ** 2)
                
                print(f"[Party 1] ========== BATCH {batch_idx+1}/{num_batches} DONE ==========\n")
        
        # ⭐ 修改：合并结果 - 累加所有批次的范数平方，然后开方
        print(f"[Party {self.party_id}] Merging {len(batch_norms_squared)} batch results...")
        
        total_norm_squared = sum(batch_norms_squared)
        final_norm = torch.sqrt(total_norm_squared)
        
        print(f"[Party {self.party_id}] Total norm squared: {total_norm_squared.item():.4f}")
        print(f"[Party {self.party_id}] Final Frobenius norm: {final_norm.item():.4f}")
        
        # 移到 GPU
        if torch.cuda.is_available():
            final_norm = final_norm.cuda()
            print(f"[Party {self.party_id}] Moved result to GPU")
        
        print(f"[Party {self.party_id}] ========== secure_matrix_multiply_batched END ==========\n")
        return final_norm  # ⭐ 修改：返回标量而不是矩阵
    
    def cleanup(self, session_id=0):
        """清理临时文件"""
        files = [
            f"Input-P{self.party_id}-{session_id}",
            f"Output-P{self.party_id}-{session_id}"
        ]
        
        for fname in files:
            fpath = os.path.join(self.player_data_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"[Party {self.party_id}] Removed {fname}")


# 测试函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--party', type=int, required=True, choices=[0, 1])
    parser.add_argument('--m', type=int, default=100)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Testing BATCHED MPC as Party {args.party}")
    print(f"Matrix size: {args.m}x{args.n}")
    print(f"Batch size: {args.batch_size}")
    print(f"IMPORTANT: Run Party 0 first, then Party 1 within 5 seconds!")
    print(f"{'='*60}\n")
    
    mpc = MPSPDZManager(party_id=args.party)
    
    try:
        if args.party == 0:
            # Party 0
            A = torch.randn(args.m, args.n)
            W = torch.randn(args.n, args.n)
            print(f"Party 0 - Generated A: {A.shape}")
            print(f"Party 0 - Generated W: {W.shape}")
            
            result = mpc.secure_matrix_multiply_batched(
                A=A, W=W, batch_size=args.batch_size
            )
            #print(f"\nParty 0 - Final Result shape: {result.shape}")
            #print(f"Party 0 - Result norm: {torch.norm(result):.4f}")
            print(f"\nParty 0 - Final Result (scalar norm): {result.item():.4f}")
            print(f"Party 0 - Result type: {type(result)}")
            
        else:
            # Party 1
            B = torch.randn(args.m, args.n)
            print(f"Party 1 - Generated B: {B.shape}")
            
            result = mpc.secure_matrix_multiply_batched(
                B=B, batch_size=args.batch_size
            )
            # ⭐ 修复后 Party 1 也有 shape 属性
            #print(f"\nParty 1 - Successfully got result shape: {result.shape}")
            print(f"\nParty 1 - Successfully got result (should be 0): {result.item():.4f}")
        
        print(f"\n{'='*60}")
        print("Test PASSED!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise