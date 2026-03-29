# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

GPU SpGEMM benchmarking suite comparing tile-based sparse matrix-matrix multiplication implementations on NVIDIA GPUs. Currently benchmarks **TileSpGEMM** (PPoPP '22) and **GPU_SPGEMM** (saltsystemslab) on the SuiteSparse 18-matrix dataset.

## Build Commands

### TileSpGEMM (CUDA 11.8, Makefile)
```bash
module load cuda/11.8.0
cd TileSpGEMM/src && make
# Run: ./test -d 0 -aat 0 matrix.mtx  (C=A^2)
# Run: ./test -d 0 -aat 1 matrix.mtx  (C=AA^T)
```
**Must build on a GPU node** — the binary links against CUDA runtime. Uses multi-arch: sm_75 (2080Ti), sm_80 (A100), sm_86 (3090/A6000).

### GPU_SPGEMM (CUDA 12.5, CMake, requires GCC 13+)
```bash
module load gcc/13.1.0 cuda/12.5.0 cmake/3.26.0
cd GPU_SPGEMM && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
# Run: ./spgemm_raw_v2 -m matrix.mtx -d 0 -a 0  (C=A^2)
```
**Must build on a GPU node** — CMake auto-detects GPU architecture via `EvalGpuArchs.cmake`.

## Running Benchmarks

All benchmarks must run via SLURM (never on login nodes). Submit scripts with `sbatch`:

- `run_benchmark.sh` — TileSpGEMM on all 18 matrices (A^2 + AA^T), rebuilds on compute node
- `run_gpuspgemm_bench.sh` — GPU_SPGEMM on all 18 matrices (A^2 only), builds + runs

Both scripts target A100 via `notchpeak-gpu-guest` partition. Matrices are stored at `/scratch/general/vast/u1446071/spgemm_matrices/`. Run `download_matrices.sh` on the login node first if matrices are missing.

## Architecture

### Submodules
- **TileSpGEMM/** — PPoPP '22 tiled SpGEMM (upstream: SuperScientificSoftwareLaboratory/TileSpGEMM). Uses CUDA 11.8 due to bf16 header bug in 12.x.
- **GPU_SPGEMM/** — Tile-based raw CUDA kernel SpGEMM (fork: jianjianh1/GPU_SPGEMM). CPU correctness verification is disabled (`if (false)`) for benchmarking speed.

### Key differences between implementations
- **TileSpGEMM**: CSR→tile conversion + bitmask-based symbolic SpGEMM on CPU, then GPU numeric. Has O(tilem×tilen/32) CPU bitmask overhead that causes DNF on matrices with >500K rows (mc2depi, webbase-1M, af_shell10).
- **GPU_SPGEMM**: All-GPU pipeline (tile structure, symbolic, numeric). Uses CUB for scans. Handles large-row matrices fine but has slow CPU correctness verification (disabled).

### Data types
- TileSpGEMM: **double** precision (VALUE_TYPE macro)
- GPU_SPGEMM: **float** (single precision)

### TileSpGEMM quirks
- The binary writes CSV results to `../data/` relative to CWD — must run from `TileSpGEMM/src/` or it segfaults (NULL fprintf).
- AA^T operation is skipped for symmetric matrices ("matrix AAT does not do symmetric matrix. Exit.").
- Tile size is fixed at 16×16.

## Results

All results are in `results/`. Key file: `comparison_a100.csv` has the head-to-head comparison.

## Benchmark Matrices

18 matrices from SuiteSparse (Table 2 of TileSpGEMM paper). Downloaded via `download_matrices.sh` which uses correct Group/Name mappings (e.g., Williams/pdb1HYS, QCD/conf5_4-8x8-05, QY/case39). Stored in MatrixMarket (.mtx) format.
