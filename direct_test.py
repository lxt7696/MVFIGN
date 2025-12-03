# direct_test.py - 直接测试 MPC（不通过 mpc_utils.py）

import subprocess
import os
import time

mpspdz_root = "/home/lxt/project/MVFIGN/MP-SPDZ"
player_data_dir = os.path.join(mpspdz_root, "Player-Data")

print("="*60)
print("Direct MPC Test")
print("="*60)

# 1. 编译
print("\n1. Compiling...")
compile_cmd = f"cd {mpspdz_root} && ./compile.py -R 64 secure_matmul"
result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print("Compilation failed!")
    print(result.stderr)
    exit(1)
print("✓ Compilation successful")

# 2. 创建输入文件
print("\n2. Creating input files...")

# Party 0: A (4x4) + W (4x4) = 32 values
party0_input = os.path.join(player_data_dir, "Input-P0-0")
with open(party0_input, 'w') as f:
    # A matrix
    for i in range(16):
        f.write(f"{(i+1)*0.1}\n")
    # W matrix
    for i in range(16):
        f.write(f"{(i+1)*0.05}\n")
print(f"✓ Created {party0_input}")

# Party 1: B (4x4) = 16 values
party1_input = os.path.join(player_data_dir, "Input-P1-0")
with open(party1_input, 'w') as f:
    for i in range(16):
        f.write(f"{(i+1)*0.08}\n")
print(f"✓ Created {party1_input}")

# 3. 启动 Party 0
print("\n3. Starting Party 0 (server)...")
cmd0 = f"cd {mpspdz_root} && ./semi2k-party.x -p 0 -N 2 -h localhost secure_matmul"
print(f"   Command: {cmd0}")

proc0 = subprocess.Popen(
    cmd0,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print(f"   PID: {proc0.pid}")

# 4. 等待服务器启动
print("\n4. Waiting for server to start...")
time.sleep(3)

# 5. 启动 Party 1
print("\n5. Starting Party 1 (client)...")
cmd1 = f"cd {mpspdz_root} && ./semi2k-party.x -p 1 -N 2 -h localhost secure_matmul"
print(f"   Command: {cmd1}")

proc1 = subprocess.Popen(
    cmd1,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print(f"   PID: {proc1.pid}")

# 6. 等待完成
print("\n6. Waiting for completion (max 60 seconds)...")

try:
    stdout0, stderr0 = proc0.communicate(timeout=60)
    stdout1, stderr1 = proc1.communicate(timeout=60)
except subprocess.TimeoutExpired:
    print("✗ Timeout! Killing processes...")
    proc0.kill()
    proc1.kill()
    stdout0, stderr0 = proc0.communicate()
    stdout1, stderr1 = proc1.communicate()

# 7. 显示结果
print("\n" + "="*60)
print("PARTY 0 OUTPUT")
print("="*60)
print("STDOUT:")
print(stdout0)
print("\nSTDERR:")
print(stderr0)

print("\n" + "="*60)
print("PARTY 1 OUTPUT")
print("="*60)
print("STDOUT:")
print(stdout1)
print("\nSTDERR:")
print(stderr1)

print("\n" + "="*60)
print(f"Party 0 return code: {proc0.returncode}")
print(f"Party 1 return code: {proc1.returncode}")
print("="*60)