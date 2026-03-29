#!/bin/bash
#SBATCH --job-name=tilespgemm_bench
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
mkdir -p "$RESULTS_DIR"

# Rebuild on compute node to ensure correct linking
echo "=== Building on compute node ==="
cd "$SRCDIR"
make clean 2>/dev/null || true
make
cd -

BINARY="$SRCDIR/test"
# The binary writes CSV results to ../data/ relative to CWD, so run from src/
cd "$SRCDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/tilespgemm_a100_${TIMESTAMP}.txt"

echo "=== TileSpGEMM Benchmark ===" | tee "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
echo "Host: $(hostname)" | tee -a "$RESULT_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$RESULT_FILE"
echo "CUDA: $(nvcc --version | tail -1)" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

MATRICES=(
    pdb1HYS
    consph
    cant
    pwtk
    rma10
    conf5_4-8x8-05
    shipsec1
    mac_econ_fwd500
    mc2depi
    cop20k_A
    scircuit
    webbase-1M
    af_shell10
    pkustk12
    SiO2
    case39
    TSOPF_FS_b300_c2
    gupta3
)

for mat in "${MATRICES[@]}"; do
    MTX="$MATRIX_DIR/$mat.mtx"
    if [ ! -f "$MTX" ]; then
        echo "[SKIP] $mat — file not found" | tee -a "$RESULT_FILE"
        continue
    fi

    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=A^2: $mat ===" | tee -a "$RESULT_FILE"
    timeout 120 stdbuf -oL "$BINARY" -d 0 -aat 0 "$MTX" 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat A^2 failed (exit=$RC)" | tee -a "$RESULT_FILE"

    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=AA^T: $mat ===" | tee -a "$RESULT_FILE"
    timeout 120 stdbuf -oL "$BINARY" -d 0 -aat 1 "$MTX" 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat AA^T failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
