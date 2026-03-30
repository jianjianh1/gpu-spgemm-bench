#!/bin/bash
#SBATCH --job-name=gpuspgemm_af
#SBATCH --time=01:00:00
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

BINARY="$(pwd)/GPU_SPGEMM/build/spgemm_raw_v2"
MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"

echo "=== GPU_SPGEMM af_shell10 ==="
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

echo "=== C=A^2: af_shell10 ==="
timeout 1800 stdbuf -oL "$BINARY" -m "$MATRIX_DIR/af_shell10.mtx" -d 0 -a 0 2>&1
