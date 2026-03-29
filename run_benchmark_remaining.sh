#!/bin/bash
#SBATCH --job-name=tilespgemm_remaining
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-bench-%j.out
#SBATCH -e slurm-bench-%j.err

set -uo pipefail

module load cuda/11.8.0

SRCDIR="$(pwd)/TileSpGEMM/src"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
RESULTS_DIR="$(pwd)/results"
BINARY="$SRCDIR/test"
cd "$SRCDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/tilespgemm_a100_remaining_${TIMESTAMP}.txt"

echo "=== TileSpGEMM Benchmark (remaining matrices) ===" | tee "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
echo "Host: $(hostname)" | tee -a "$RESULT_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$RESULT_FILE"
echo "Timeout: 600s per run" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

MATRICES=(mc2depi webbase-1M af_shell10)

for mat in "${MATRICES[@]}"; do
    MTX="$MATRIX_DIR/$mat.mtx"

    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=A^2: $mat ===" | tee -a "$RESULT_FILE"
    timeout 600 stdbuf -oL "$BINARY" -d 0 -aat 0 "$MTX" 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat A^2 failed (exit=$RC)" | tee -a "$RESULT_FILE"

    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=AA^T: $mat ===" | tee -a "$RESULT_FILE"
    timeout 600 stdbuf -oL "$BINARY" -d 0 -aat 1 "$MTX" 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat AA^T failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
