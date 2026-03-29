#!/bin/bash
#SBATCH --job-name=gpuspgemm_bench
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

module load gcc/13.1.0
module load cuda/12.5.0
module load cmake/3.26.0

export CC=$(which gcc)
export CXX=$(which g++)

PROJDIR="$(pwd)"
BUILDDIR="$PROJDIR/GPU_SPGEMM/build"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
RESULTS_DIR="$PROJDIR/results"
mkdir -p "$RESULTS_DIR"

# Build on compute node (needs GPU for arch detection)
echo "=== Building GPU_SPGEMM ==="
mkdir -p "$BUILDDIR"
cd "$BUILDDIR"
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_BUILD_TYPE=Release 2>&1
make -j8 2>&1
cd "$PROJDIR"

BINARY="$BUILDDIR/spgemm_raw_v2"
if [ ! -f "$BINARY" ]; then
    echo "Build failed!"
    exit 1
fi
echo "=== Build successful ==="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/gpuspgemm_a100_${TIMESTAMP}.txt"

echo "=== GPU_SPGEMM Benchmark ===" | tee "$RESULT_FILE"
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
    timeout 600 stdbuf -oL "$BINARY" -m "$MTX" -d 0 -a 0 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat A^2 failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
