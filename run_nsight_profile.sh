#!/bin/bash
#SBATCH --job-name=nsight_prof
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-nsight-%j.out
#SBATCH -e slurm-nsight-%j.err

set -uo pipefail

module load gcc/13.1.0
module load cuda/12.5.0
module load cmake/3.26.0

export CC=$(which gcc)
export CXX=$(which g++)

PROJDIR="$(pwd)"
BUILDDIR="$PROJDIR/GPU_SPGEMM/build"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
RESULTS_DIR="$PROJDIR/results/nsight"
mkdir -p "$RESULTS_DIR"

# Build with lineinfo for NSight
echo "=== Building GPU_SPGEMM with lineinfo ==="
mkdir -p "$BUILDDIR"
cd "$BUILDDIR"
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_BUILD_TYPE=Release -DLINFO=ON 2>&1
make -j8 2>&1
cd "$PROJDIR"

BINARY="$BUILDDIR/spgemm_raw_v2"
NCU=$(which ncu)

echo "=== NSight Compute Profiling ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

MATRICES=(rgg_n_2_21_s0 Queen_4147 consph StocF-1465 cage14)

for mat in "${MATRICES[@]}"; do
    MTX="$MATRIX_DIR/$mat.mtx"
    if [ ! -f "$MTX" ]; then
        echo "[SKIP] $mat — not found"
        continue
    fi
    echo "=== Profiling: $mat ==="

    # First run without profiling to get timing
    echo "--- Baseline timing ---"
    timeout 300 "$BINARY" -m "$MTX" -d 0 -a 0 2>&1 | grep -E "GPU time|Step|nnzC|flops"

    # NSight Compute: profile all kernels
    echo "--- NSight Compute ---"
    timeout 600 "$NCU" \
        --set full \
        --export "$RESULTS_DIR/${mat}" \
        "$BINARY" -m "$MTX" -d 0 -a 0 2>&1 | tail -50

    echo ""
done

echo "=== Profiling complete ==="
ls -lh "$RESULTS_DIR"/*.ncu-rep 2>/dev/null
