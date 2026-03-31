#!/bin/bash
#SBATCH --job-name=correct_test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-correct-%j.out
#SBATCH -e slurm-correct-%j.err

set -uo pipefail
module load cuda/11.8.0

SRCDIR="$(pwd)/TileSpGEMM/src"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"

echo "=== Building TileSpGEMM (original values, CHECK_RESULT=1) ==="
cd "$SRCDIR"
make clean 2>/dev/null || true
make CUDA_INSTALL_PATH="$CUDA_HOME" \
     NVCC_FLAGS="-O3 -w -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86"
cd "$SRCDIR"

BINARY="$SRCDIR/test"

MATRICES=(cant rma10 scircuit mac_econ_fwd500 mc2depi cop20k_A conf5_4-8x8-05 case39)

for mat in "${MATRICES[@]}"; do
    MTX="$MATRIX_DIR/$mat.mtx"
    echo ""
    echo "============================================"
    echo "=== C=A^2: $mat ==="
    echo "============================================"
    timeout 300 stdbuf -oL "$BINARY" -d 0 -aat 0 "$MTX" 2>&1
    echo ""
done
