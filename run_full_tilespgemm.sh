#!/bin/bash
#SBATCH --job-name=tile_full
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-tile-full-%j.out
#SBATCH -e slurm-tile-full-%j.err

set -uo pipefail

module load cuda/11.8.0

SRCDIR="$(pwd)/TileSpGEMM/src"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"

echo "=== Building TileSpGEMM ==="
cd "$SRCDIR"
make clean 2>/dev/null || true
make CUDA_INSTALL_PATH="$CUDA_HOME" \
     NVCC_FLAGS="-O3 -w -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86"
cd "$SRCDIR"

BINARY="$SRCDIR/test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/tilespgemm_full_a100_${TIMESTAMP}.txt"

echo "=== TileSpGEMM Full Benchmark ===" | tee "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

for MTX in "$MATRIX_DIR"/*.mtx; do
    mat=$(basename "$MTX" .mtx)
    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=A^2: $mat ===" | tee -a "$RESULT_FILE"
    timeout 300 stdbuf -oL "$BINARY" -d 0 -aat 0 "$MTX" 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat A^2 failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
