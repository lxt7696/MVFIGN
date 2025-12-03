# diagnose_mpc.py - 诊断 MPC 连接问题

import subprocess
import os
import time

def check_mpspdz_installation():
    """检查 MP-SPDZ 安装"""
    print("="*60)
    print("1. Checking MP-SPDZ installation...")
    print("="*60)
    
    mpspdz_root = "/home/lxt/project/MVFIGN/MP-SPDZ"
    
    files_to_check = [
        "compile.py",
        "semi2k-party.x",
        "Programs/Source/secure_matmul.mpc"
    ]
    
    for f in files_to_check:
        path = os.path.join(mpspdz_root, f)
        exists = os.path.exists(path)
        print(f"  {f}: {'✓ EXISTS' if exists else '✗ MISSING'}")
    
    print()

def check_port_availability():
    """检查端口可用性"""
    print("="*60)
    print("2. Checking port 5000 availability...")
    print("="*60)
    
    result = subprocess.run(
        "netstat -tuln | grep 5000 || echo 'Port 5000 is free'",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print()

def test_mpc_program():
    """测试 MPC 程序本身"""
    print("="*60)
    print("3. Testing MPC program compilation...")
    print("="*60)
    
    mpspdz_root = "/home/lxt/project/MVFIGN/MP-SPDZ"
    
    # 编译
    compile_cmd = f"cd {mpspdz_root} && ./compile.py -R 64 secure_matmul"
    print(f"Running: {compile_cmd}")
    
    result = subprocess.run(
        compile_cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed:")
        print(result.stderr)
    
    print()

def check_mpc_output():
    """检查 MPC 程序的输出格式"""
    print("="*60)
    print("4. Checking secure_matmul.mpc content...")
    print("="*60)
    
    mpspdz_root = "/home/lxt/project/MVFIGN/MP-SPDZ"
    mpc_file = os.path.join(mpspdz_root, "Programs/Source/secure_matmul.mpc")
    
    if os.path.exists(mpc_file):
        with open(mpc_file, 'r') as f:
            content = f.read()
            if '=== RESULTS ===' in content:
                print("✓ Output markers found in MPC file")
            else:
                print("✗ Output markers NOT found - this is the problem!")
                print("\nCurrent content:")
                print(content[-500:])  # 显示最后500字符
    else:
        print("✗ MPC file not found!")
    
    print()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MPC DIAGNOSIS TOOL")
    print("="*60 + "\n")
    
    check_mpspdz_installation()
    check_port_availability()
    test_mpc_program()
    check_mpc_output()
    
    print("="*60)
    print("Diagnosis complete!")
    print("="*60)