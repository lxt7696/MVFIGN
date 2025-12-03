#!/usr/bin/env python3
import subprocess
import sys

party_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mpspdz_root = "/home/lxt/project/MVFIGN/MP-SPDZ"

print(f"[Party {party_id}] Starting MPC test...")

# 1. 编译程序（只在 Party 0 编译）
if party_id == 0:
    print("[Party 0] Compiling MPC program...")
    compile_result = subprocess.run(
        f"cd {mpspdz_root} && ./compile.py secure_matmul",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if compile_result.returncode != 0:
        print(f"Compilation failed: {compile_result.stderr}")
        sys.exit(1)
    print("[Party 0] Compilation successful!")

# 2. 等待对方准备好
import time
if party_id == 0:
    print("[Party 0] Waiting for Party 1...")
    time.sleep(3)
else:
    print("[Party 1] Connecting to Party 0...")

# 3. 运行 MPC
print(f"[Party {party_id}] Running MPC computation...")
run_result = subprocess.run(
    f"cd {mpspdz_root} && ./semi2k-party.x -p {party_id} secure_matmul",
    shell=True,
    capture_output=True,
    text=True,
    timeout=30
)

if run_result.returncode == 0:
    print(f"[Party {party_id}] Success!")
    print(run_result.stdout)
else:
    print(f"[Party {party_id}] Failed!")
    print(run_result.stderr)