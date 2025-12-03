#!/bin/bash
# Automated test script - runs both parties simultaneously

echo "=============================================="
echo "Automated MPC Test - Both Parties"
echo "=============================================="
echo ""

# Set paths
MPSPDZ_ROOT="/home/lxt/project/MVFIGN/MP-SPDZ"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cleanup old files
echo "Cleaning up old files..."
rm -f "$MPSPDZ_ROOT/Player-Data/Input-P0-0" "$MPSPDZ_ROOT/Player-Data/Input-P1-0"
rm -f "$MPSPDZ_ROOT/Player-Data/P0-O0" "$MPSPDZ_ROOT/Player-Data/P1-O0"

# Copy files
echo "Copying MPC program and utilities..."
cp "$SCRIPT_DIR/secure_matmul_fixed.mpc" "$MPSPDZ_ROOT/Programs/Source/"
cp "$SCRIPT_DIR/mpc_utils_fixed.py" "$MPSPDZ_ROOT/"

# Navigate to MP-SPDZ
cd "$MPSPDZ_ROOT"

# Compile the program once
echo ""
echo "Compiling MPC program..."
./compile.py -R 64 secure_matmul_fixed

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo ""
echo "Starting Party 1 in background..."
python3 mpc_utils_fixed.py --party 1 --m 4 --n 4 > party1.log 2>&1 &
PARTY1_PID=$!

# Wait a moment for Party 1 to initialize
sleep 2

echo "Starting Party 0..."
python3 mpc_utils_fixed.py --party 0 --m 4 --n 4

# Wait for Party 1 to finish
echo ""
echo "Waiting for Party 1 to complete..."
wait $PARTY1_PID

# Show results
echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
echo ""
echo "Party 0 Output (above)"
echo ""
echo "Party 1 Output:"
cat party1.log
echo ""
echo "=============================================="