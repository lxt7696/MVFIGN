# Fixed MPC utility class for MP-SPDZ integration

import subprocess
import numpy as np
import torch
import os
import time

class MPSPDZManager:
    """MP-SPDZ Secure Computation Manager"""
    
    def __init__(self, party_id, mpspdz_root="/home/lxt/project/MVFIGN/MP-SPDZ"):
        self.party_id = party_id
        self.mpspdz_root = mpspdz_root
        self.player_data_dir = os.path.join(mpspdz_root, "Player-Data")
        os.makedirs(self.player_data_dir, exist_ok=True)
        
        print(f"[Party {party_id}] MPC Manager initialized")
        print(f"  MP-SPDZ Root: {mpspdz_root}")
        
    def write_inputs(self, data_dict, session_id=0):
        """Write data to MP-SPDZ input file"""
        input_file = os.path.join(
            self.player_data_dir, 
            f"Input-P{self.party_id}-{session_id}"
        )
        
        with open(input_file, 'w') as f:
            for key, tensor in data_dict.items():
                if tensor is None:
                    continue
                    
                if isinstance(tensor, torch.Tensor):
                    data = tensor.detach().cpu().numpy()
                else:
                    data = tensor
                
                flat_data = data.flatten()
                for val in flat_data:
                    f.write(f"{float(val)}\n")
        
        print(f"[Party {self.party_id}] Wrote {len(flat_data)} values to {input_file}")
    
    def read_outputs_from_file(self, shape, session_id=0, timeout=60):
        """Read results from MP-SPDZ output file"""
        output_file = os.path.join(
            self.player_data_dir,
            f"P{self.party_id}-O{session_id}"
        )
        
        wait_time = 0
        while not os.path.exists(output_file) and wait_time < timeout:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(
                f"Output file not found: {output_file}\n"
                f"Make sure both parties are running!"
            )
        
        with open(output_file, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        
        result = np.array(values).reshape(shape)
        print(f"[Party {self.party_id}] Read output shape {result.shape}")
        
        return torch.from_numpy(result).float()
    
    def compile_program(self, program_name="secure_matmul_fixed"):
        """Compile MPC program"""
        compile_cmd = f"cd {self.mpspdz_root} && ./compile.py -R 64 {program_name}"
        
        print(f"[Party {self.party_id}] Compiling MPC program...")
        
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Compilation error: {result.stderr}")
            raise RuntimeError("MPC program compilation failed")
        
        print(f"[Party {self.party_id}] Compilation successful")
    
    def run_party(self, program_name="secure_matmul_fixed", 
                  host="localhost", port_base=5000, timeout=120):
        """Run MPC party"""
        run_cmd = (
            f"cd {self.mpspdz_root} && "
            f"./semi2k-party.x -p {self.party_id} "
            f"-h {host} -pn {port_base} "
            f"{program_name}"
        )
        
        print(f"[Party {self.party_id}] Running MPC computation...")
        print(f"  IMPORTANT: Party {1-self.party_id} must also be running!")
        
        try:
            result = subprocess.run(run_cmd, shell=True, capture_output=True, 
                                  text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"Execution error: {result.stderr}")
                raise RuntimeError("MPC execution failed")
            
            print(f"[Party {self.party_id}] MPC computation completed")
            return result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"[Party {self.party_id}] ERROR: Timeout!")
            print(f"  Make sure Party {1-self.party_id} is running!")
            raise
    
    def secure_matrix_multiply(self, A=None, B=None, W=None, 
                               compile_once=True, session_id=0):
        """Secure computation of (A - B) * W"""
        if self.party_id == 0:
            if A is None or W is None:
                raise ValueError("Party 0 must provide A and W")
            m, n = A.shape
            self.write_inputs({'A': A, 'W': W}, session_id)
        else:
            if B is None:
                raise ValueError("Party 1 must provide B")
            m, n = B.shape
            self.write_inputs({'B': B}, session_id)
        
        if not compile_once or session_id == 0:
            self.compile_program("secure_matmul_fixed")
        
        self.run_party("secure_matmul_fixed")
        result = self.read_outputs_from_file((m, n), session_id)
        
        return result
    
    def cleanup(self, session_id=0):
        """Clean up temporary files"""
        files_to_remove = [
            f"Input-P{self.party_id}-{session_id}",
            f"P{self.party_id}-O{session_id}"
        ]
        
        for filename in files_to_remove:
            filepath = os.path.join(self.player_data_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        print(f"[Party {self.party_id}] Cleaned up session {session_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MP-SPDZ MPC')
    parser.add_argument('--party', type=int, required=True, choices=[0, 1])
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--n', type=int, default=4)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Testing MPC as Party {args.party}")
    print(f"{'='*60}\n")
    
    mpc = MPSPDZManager(party_id=args.party)
    
    if args.party == 0:
        A = torch.randn(args.m, args.n)
        W = torch.randn(args.n, args.n)
        print(f"Party 0 - Matrix A:\n{A}")
        print(f"\nParty 0 - Weight W:\n{W}")
        print("\nWaiting for Party 1...")
        result = mpc.secure_matrix_multiply(A=A, W=W)
    else:
        B = torch.randn(args.m, args.n)
        print(f"Party 1 - Matrix B:\n{B}")
        print("\nWaiting for Party 0...")
        result = mpc.secure_matrix_multiply(B=B)
    
    print(f"\n{'='*60}")
    print(f"Party {args.party} Result:")
    print(result)
    print(f"{'='*60}\n")