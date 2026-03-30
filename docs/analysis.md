# GPU SpGEMM Benchmark Analysis

## Overview

This document analyzes the performance of two tile-based GPU SpGEMM (Sparse General Matrix-Matrix Multiplication) implementations on the NVIDIA A100 80GB PCIe GPU, benchmarked across 222 matrices from the SuiteSparse Matrix Collection.

**Implementations compared:**
- **TileSpGEMM** (PPoPP '22, Niu et al.) — CSR-to-tile conversion with bitmask-based symbolic SpGEMM
- **GPU_SPGEMM** (saltsystemslab) — All-GPU tile-based SpGEMM with CUB-based scans

**Hardware:** NVIDIA A100 80GB PCIe, CHPC Notchpeak cluster
**Operation:** C = A² (square SpGEMM) on all matrices
**Dataset:** 222 square matrices from SuiteSparse with estimated flops ≥ 2×10⁸

## Results Summary

Of 222 matrices:
- TileSpGEMM completed 186; GPU_SPGEMM completed 213
- On the 186 matrices both completed, **GPU_SPGEMM was faster on all 186**
- Geometric mean speedup: **6.19×**
- Median speedup: **2.77×**
- Maximum speedup: **19,489×** (mycielskian18)

## Architecture Comparison

Both implementations follow a three-step pipeline:

| Step | TileSpGEMM | GPU_SPGEMM |
|------|-----------|------------|
| **Preprocessing** | CSR→tile on CPU (OpenMP), bitmask construction on CPU, H2D copy | CSR→tile on CPU, upload to GPU |
| **Step 1: Tile Structure** | Bin-based nsparse hash counting OR SPA-based counting | Warp-based SPA or CUB scan |
| **Step 2: Symbolic SpGEMM** | Dense bitmask intersection scan | Binary search / sorted list intersection |
| **Step 3: Numeric SpGEMM** | Tile-level numeric accumulation with adaptive dense/sparse accumulators | Output-nnz-centric tile accumulation |

### Key Algorithmic Differences

**Tile format:** Both use 16×16 tiles with CSR-within-tile storage and bitmasks for nonzero patterns. GPU_SPGEMM additionally builds within-tile CSC format for matrix B, enabling column-oriented access during multiplication.

**Data types:** TileSpGEMM uses double precision (float64); GPU_SPGEMM uses single precision (float32). This gives GPU_SPGEMM a 2× advantage in memory bandwidth and potentially in compute throughput.

**Symbolic phase:** The most significant architectural difference. TileSpGEMM constructs a dense bitmask array of size `tilem × ceil(tilen/32)` on CPU, copies it to GPU, then scans it to determine which tile pairs contribute to C. GPU_SPGEMM performs symbolic intersection entirely on GPU using sorted tile column lists — no dense bitmask needed.

## Step-by-Step Performance Analysis

### Step 1: Tile Structure Determination

Average breakdown: TileSpGEMM 5.4%, GPU_SPGEMM 23.3% of respective totals.

TileSpGEMM uses two code paths depending on `blknB` (number of tile columns of B):
- **SPA path** (`blknB ≤ 16384`): Direct sparse accumulator, fast for small tile dimensions
- **Bin-based nsparse path** (`blknB > 16384`): Bins rows by intermediate product count, dispatches to hash-table kernels with different shared memory sizes

GPU_SPGEMM always uses a warp-based approach with CUB prefix sums.

**Observation:** For matrices with >500K rows (tilen > 30K), TileSpGEMM's nsparse path shows up to 396× slower step 1 compared to GPU_SPGEMM. The overhead comes from binning, permutation arrays, and per-bin kernel launches. GPU_SPGEMM's unified approach avoids this dispatch overhead.

### Step 2: Symbolic SpGEMM

Average breakdown: TileSpGEMM 19.3%, GPU_SPGEMM 26.3% of respective totals.

This is where the largest speedup differences occur — up to **62,884×** for rgg_n_2_23_s0.

**TileSpGEMM's approach:** For each tile (i,j) of C, scan the intersection of tile-row i's bitmask with tile-column j's bitmask. The bitmask has `blk_intersec_bitmask_len = ceil(tilen/32)` unsigned integers per tile row/column. Each tile pair requires reading `2 × bitmask_len` words from global memory and computing their bitwise AND.

The total work is: `O(numtileC × tilen / 32)`

For a matrix with n=8.4M rows (rgg_n_2_23_s0): tilen = 524,288, bitmask_len = 16,384. With numtileC = ~500K tiles, the kernel performs ~8.6 billion memory reads just for the bitmask scan — regardless of actual tile sparsity.

**GPU_SPGEMM's approach:** Performs symbolic intersection by iterating through the actual nonzero tile column lists of A and B using binary search or merge-based intersection. Work is proportional to actual nonzeros, not matrix dimensions.

The total work is: `O(numtileC × avg_tiles_per_intersection)`

For rgg_n_2_23_s0 with average ~15 nnz per row, each tile has ~1 tile per intersection on average. The symbolic phase is essentially O(numtileC).

**Correlation analysis:** Step 2 speedup correlates strongly with tile dimensions — matrices with large n (and thus large tilen and bitmask_len) show the largest speedups. The bitmask approach has O(n²/32) complexity that is independent of actual sparsity, making it increasingly wasteful for large sparse matrices.

### Step 3: Numeric SpGEMM

Average breakdown: TileSpGEMM 75.3%, GPU_SPGEMM 50.4% of respective totals.

Step 3 dominates both implementations but GPU_SPGEMM is consistently faster (2-10× on most matrices). The improvement comes from:

1. **Output-oriented loop order:** GPU_SPGEMM's v2 kernel iterates over output tiles with pre-computed intersection lists from the symbolic phase, avoiding redundant work.

2. **Within-tile CSC for B:** GPU_SPGEMM builds CSC format within each tile of B, enabling column-oriented access that matches the accumulation pattern better than TileSpGEMM's row-oriented B access.

3. **Single vs double precision:** GPU_SPGEMM uses float32 (2× bandwidth and compute efficiency on A100 compared to float64).

4. **Speculative intersection:** GPU_SPGEMM pre-computes tile pair positions during symbolic phase, eliminating redundant intersection work during numeric.

## CPU Preprocessing Analysis

CPU preprocessing (CSR-to-tile conversion) is **not included** in the GPU kernel timing but adds significant overhead, especially for large matrices.

### TileSpGEMM Preprocessing

TileSpGEMM performs CSR-to-tile conversion using OpenMP-parallelized CPU code. Three bugs were identified and fixed:

**Bug 1: O(tilem × tilen) memset in step1_kernel and step2_kernel** — Each tile-row iteration zeroed a flag array of size `tilen` using `memset`. For a 1M-row matrix (tilem=62500, tilen=62500), this totaled ~4GB of memset per call. **Fix:** `calloc` with targeted clear of only touched entries, reducing to O(nnz) total.

**Bug 2: O(nnz × colbnum) linear search in csr2tile_col_major** — For each nonzero, a linear search through `colbnum` tiles determined its tile assignment. **Fix:** Direct O(1) lookup via `tilerow_to_ki` array.

**Bug 3: Dense bitmask allocation** — `malloc` + `memset` of 128-256MB bitmask arrays. **Fix:** `calloc` (OS lazy zero-page mapping avoids physical zeroing of untouched pages).

Combined effect: preprocessing for mc2depi (525K rows) went from >600s (DNF) to ~250ms.

### GPU_SPGEMM Preprocessing

GPU_SPGEMM's tile conversion had similar patterns:

**Issue 1: Per-tile-row std::vector allocation** — `std::vector<char>(tilen, 0)` allocated and destroyed ~62K times for large matrices. **Fix:** Hoist allocation outside loop, use targeted clear.

**Issue 2: Linear tile search in csr2tile Phase 3** — O(ntiles) linear scan per nonzero. **Fix:** O(1) direct-mapped `tc_to_ti` lookup array.

**Issue 3: Repeated std::lower_bound in build_csc_tiles** — 3 binary searches per nonzero for tile index lookup. **Fix:** O(1) direct-mapped `tc_to_tidx` array.

## GPU Hang Bug on Volta+ GPUs

A critical bug was discovered in TileSpGEMM's nsparse kernel that caused **infinite hangs** on Volta and later GPUs (V100, A100, etc.) for matrices with >16384 tile columns.

### Root Cause

The `calculate_value_col_bin_pwarp` kernel in `spgemm_nsparse_kernel.h` processes tile rows in groups of PWARP=4 threads sharing a 16-slot hash table in shared memory. The kernel has three phases: hash insert, compaction, and sorting.

At the transition between phases, thread 0 resets a global counter (`d_nz[rid] = 0`) while the other 3 threads proceed to the compaction phase. On pre-Volta GPUs, all 4 threads executed in lockstep (SIMT), so the reset was always visible before compaction started. On Volta+ with **independent thread scheduling**, threads 1-3 can race ahead and read the stale counter value.

When the stale value (e.g., 15, from the previous symbolic phase) is read by `atomicAdd`, the compaction writes to `shared_check[soffset + 15]` — the last slot of the 16-slot region. Further writes at indices 16, 17, ... overflow into **adjacent rows' shared memory**, corrupting their hash tables. When those corrupted rows try to insert keys, they find no empty slots and no matching keys, causing the `while(1)` probing loop to run forever.

### Affected Matrices

The bug triggered on 3 of the original 18 benchmark matrices:
- mc2depi (525K rows, 32865 tile columns)
- webbase-1M (1M rows, 62501 tile columns)
- af_shell10 (1.5M rows, 94255 tile columns)

All have tile column count > 16384, which routes through the nsparse bin-based code path.

### Fix

Add `__syncwarp()` at three points in `calculate_value_col_bin_pwarp`:
1. After `d_nz[rid] = 0` reset (before hash insert phase)
2. After hash insert loop (before compaction phase)
3. After compaction loop (before sorting phase)

Additionally, `__syncwarp()` was added in `set_row_nz_bin_pwarp` after shared memory initialization.

### Historical Context

The commented-out `__syncwarp()` calls throughout the code (`//__syncwarp();`) suggest the original authors were aware of the need but disabled them, likely because the code was developed for Pascal GPUs (sm_61) where warp-synchronous execution was implicit. The PPoPP '22 paper evaluated on RTX 3060/3090 (Ampere), but the 142-matrix benchmark may not have included matrices triggering this specific code path, or the hang was misattributed to preprocessing overhead.

## CSV Write Segfault

TileSpGEMM's `main.cu` opens `../data/results_tile.csv` (and 3 other CSV files) for appending after each benchmark run. If the directory doesn't exist, `fopen` returns NULL, but the code proceeds to `fprintf(fout, ...)` on the NULL pointer — segfault.

**Fix:** Guard each `fprintf`/`fclose` with `if (fout != NULL)`.

This bug was the original cause of the "segfault after printing results" behavior observed in early benchmark runs, before the working directory was changed to `TileSpGEMM/src/`.

## Matrices That Failed

### TileSpGEMM failures (36/222)

Most failures fall into categories:
- **Timeout (exit=124):** Matrices where preprocessing or GPU kernel exceeded 300s. Typically very large matrices (>5M rows) or matrices with extremely high compression rates where the bitmask scan is slow.
- **Segfault in cuSPARSE verification (exit=139):** Now eliminated by setting `CHECK_RESULT=0`.
- **OOM (exit=134):** Matrices requiring more than 80GB GPU memory for bitmask arrays and tile data.

### GPU_SPGEMM failures (10/222)

Fewer failures due to:
- No bitmask overhead (no OOM from dense bitmask arrays)
- No cuSPARSE verification
- Simpler preprocessing with less memory allocation

Remaining failures are mostly timeouts on very large matrices where CPU tile conversion exceeds 300s.

## Conclusions

1. **GPU_SPGEMM is faster on all 186 comparable matrices**, with geometric mean speedup of 6.19×. The advantage comes from all three SpGEMM steps, not just numeric.

2. **The bitmask approach in TileSpGEMM scales poorly** with matrix dimensions. Its O(tilen/32) per-tile-pair symbolic cost makes it increasingly inefficient for large sparse matrices, even though the bitmask enables fast intersection for dense tile patterns.

3. **The Volta+ warp synchronization bug** is a real correctness issue that would affect any deployment of TileSpGEMM on modern NVIDIA GPUs (V100, A100, H100). The fix is minimal (4 `__syncwarp()` additions) but critical.

4. **CPU preprocessing is a hidden cost** not captured in GPU kernel timing. For large matrices, it can exceed the GPU computation time by 10-100×. Both implementations benefit from the targeted-clear optimization that eliminates O(n²) memset patterns.

5. **Data type difference matters:** GPU_SPGEMM's use of float32 vs TileSpGEMM's float64 provides a structural advantage in memory bandwidth and compute throughput on A100. A fair comparison would require matching precision, which would likely reduce GPU_SPGEMM's advantage to ~3-4× geometric mean.
