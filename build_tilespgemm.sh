#!/bin/bash
#SBATCH --job-name=build_tilespgemm
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH -o slurm-build-%j.out
#SBATCH -e slurm-build-%j.err

set -euo pipefail

module load cuda/11.8.0

echo "=== CUDA: $(nvcc --version | tail -1) ==="
echo "=== CUDA_HOME: $CUDA_HOME ==="
nvidia-smi

cd TileSpGEMM/src
make clean 2>/dev/null || true
make
echo "=== Build successful ==="
ls -la test
