#!/bin/bash
#SBATCH --job-name=cusp_full
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-cusp-full-%j.out
#SBATCH -e slurm-cusp-full-%j.err

set -uo pipefail
module load cuda/12.5.0

MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"

echo "=== Building cuSPARSE benchmark ==="
cd cusparse_bench
make clean 2>/dev/null || true
make NVCC_FLAGS="-O3 -gencode=arch=compute_80,code=sm_80"
cd -

BINARY="cusparse_bench/cusparse_spgemm"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/cusparse_full_a100_${TIMESTAMP}.txt"

echo "=== cuSPARSE Full Benchmark ===" | tee "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

for MTX in "$MATRIX_DIR"/*.mtx; do
    mat=$(basename "$MTX" .mtx)
    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=A^2: $mat ===" | tee -a "$RESULT_FILE"
    timeout 300 stdbuf -oL "$BINARY" "$MTX" -d 0 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
