#!/bin/bash
#SBATCH --job-name=gpu_full
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --qos=soc-gpu-np
#SBATCH -o slurm-gpu-full-%j.out
#SBATCH -e slurm-gpu-full-%j.err

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/gpuspgemm_full_a100_${TIMESTAMP}.txt"

echo "=== GPU_SPGEMM Full Benchmark ===" | tee "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

for MTX in "$MATRIX_DIR"/*.mtx; do
    mat=$(basename "$MTX" .mtx)
    echo "" | tee -a "$RESULT_FILE"
    echo "=== C=A^2: $mat ===" | tee -a "$RESULT_FILE"
    timeout 300 stdbuf -oL "$BINARY" -m "$MTX" -d 0 -a 0 2>&1 | tee -a "$RESULT_FILE"
    RC=${PIPESTATUS[0]}
    [ $RC -ne 0 ] && echo "[ERROR] $mat A^2 failed (exit=$RC)" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark complete: $(date) ===" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
