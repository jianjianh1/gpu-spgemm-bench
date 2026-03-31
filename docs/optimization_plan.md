# GPU_SPGEMM Optimization Plan

## Current Performance Profile (A100, fp64)

From NSight Compute profiling on 5 representative matrices:

| Kernel | Occupancy | Registers | Bottleneck |
|---|---|---|---|
| step3_numeric_v2 | 21% | 76 | Latency-bound (FP64 MAD) |
| step2_symbolic | 64% | 40 | Memory-bound (L1 dominated) |
| step1_count/fill | 15-54% | 29-32 | Varies by matrix |
| CUB DeviceScan | 24-30% | 65 | Memory-bound |

**Current architecture of step3_numeric_kernel_v2:**
- 1 warp (32 threads) per output tile of C
- Each thread processes one row of the 16×16 tile (lane/2 = row, lane%2 = sub)
- 2 threads share one row, each handling half the A-row entries, then shuffle-reduce
- 16 TVAL registers per thread as dense accumulator for the 16 columns
- With fp64: 16 × 8 bytes = 128 bytes of accumulator per thread in registers
- Total register demand: 16 (accumulators) + ~60 (loop vars, pointers, temps) = 76

## Optimization 1: Two-Pass Accumulation (High Impact)

**Goal:** Reduce registers from 76 to ~44, increasing occupancy from 21% to ~50%.

**Approach:** Instead of accumulating all 16 columns in one pass, split into two
passes of 8 columns each.

```
// Current: 16 accumulator registers
TVAL r0=0,...,r15=0;
for each tile pair: accumulate into r0..r15
write r0..r15 to output

// Proposed: 8 accumulator registers, two passes
// Pass 1: columns 0-7
TVAL r0=0,...,r7=0;
for each tile pair:
    for each A-row entry:
        for each B-col entry:
            if (col < 8) ACCUM(col, prod)   // only accumulate low cols
write r0..r7 to output

// Pass 2: columns 8-15
TVAL r0=0,...,r7=0;
for each tile pair:
    for each A-row entry:
        for each B-col entry:
            if (col >= 8) ACCUM(col-8, prod) // only accumulate high cols
write r0..r7 to output (offset by 8)
```

**Trade-off:** 2× the tile-pair traversal work, but with ~2.4× the occupancy
(more warps to hide latency). Net speedup depends on whether the kernel is
compute-latency-bound (likely) or memory-bandwidth-bound (unlikely).

**Register estimate:** 8 (accumulators) + 36 (loop vars) = ~44 registers.
At 44 registers: 65536/44/256 = 5.8 → 5 blocks × 8 warps = 40 warps → 62.5% occupancy.

**Expected impact:** 1.5-2× speedup on Step3 for matrices where occupancy is the
bottleneck (StocF-1465, cage14).

## Optimization 2: Shared Memory Accumulator (Alternative to Opt 1)

**Goal:** Replace register-based accumulator with shared memory to free registers.

**Approach:** Each warp uses a 16×16 shared memory tile for accumulation instead
of 16 registers per thread.

```
__shared__ TVAL smem_acc[WARPS_PER_BLOCK][16][16];  // 8 warps × 256 doubles = 16 KB
// Each thread writes to smem_acc[warp][row][col] via atomicAdd
```

**Trade-off:** Shared memory bank conflicts on atomicAdd, but frees 16 registers.
With fp64, each 16×16 tile = 2 KB. With 8 warps/block, 16 KB shared memory —
fits in A100's 164 KB. Register reduction from 76 to ~48 → ~50% occupancy.

**Risk:** Shared memory atomicAdd on fp64 is emulated (not native on A100),
adding ~10 cycles per operation. For dense tiles this could be slower.

**Best for:** Sparse tiles where most columns are zero (the switch-case ACCUM
is already branch-heavy).

## Optimization 3: Fuse CUB Prefix Scans (Medium Impact)

**Problem:** CUB DeviceScan takes 128-272µs on StocF/cage14, which is 62% of
total time on cage14. Two separate kernels (DeviceScanInit + DeviceScan) with
launch overhead.

**Approach:** For arrays smaller than ~100K elements, use a single-block scan
kernel instead of CUB's multi-block approach. The tile_nnzC array has
`numtileC` elements:
- consph: 458K → CUB is fine
- cage14: 263K → CUB is fine
- StocF-1465: 629K → CUB is fine

Actually, looking at the profiling data, CUB scan takes 272µs for cage14 with
263K elements. At 2039 GB/s bandwidth and 263K × 4 bytes = 1 MB, the theoretical
time is 0.5µs. The 272µs is 544× slower than theoretical — this suggests the
scan is called multiple times (for tile_ptrC, tile_nnzC, etc.) and includes
allocation overhead.

**Approach:** Pre-allocate scan buffers once at the start instead of per-iteration.
CUB's `DeviceScan` does a temporary allocation on every call — for small arrays
this dominates runtime.

**Expected impact:** 2-5× reduction in CUB scan overhead → 10-30% total speedup
on cage14-type matrices.

## Optimization 4: Warp-Level Tile Scheduling (Medium Impact)

**Problem:** The current kernel assigns 1 warp per output tile. For matrices
with few large tiles (high compression), each warp does lots of work. For matrices
with many small tiles (low compression like cage14 with compression=6.9), each
warp does little work but there are 15K+ waves/SM.

**Approach:** Dynamic tile scheduling where warps grab tiles from a global work
queue:

```
__shared__ int next_tile;
if (threadIdx.x == 0) next_tile = atomicAdd(global_counter, 1);
__syncwarp();
int ti = next_tile;
```

This improves load balancing when tile work varies significantly.

**Expected impact:** 10-20% on matrices with high tile-count variance.

## Optimization 5: Step1 Shared Memory Reduction (Low-Medium Impact)

**Problem:** Step1 (count/fill) uses `nmasks = ceil(tilenB/32)` words of
dynamic shared memory per warp for the bitmask. For large tilenB (StocF:
6774 tile columns → nmasks=212 → 848 bytes/warp), with 4 warps/block =
3.3 KB. But for very large matrices this can exceed shared memory limits.

The profiling shows StocF uses 45.79 KB dynamic shared memory in Step1,
limiting occupancy to 15-18%.

**Approach:** For large nmasks, switch from shared memory bitmask to a
hash-set in registers or global memory. Or split the Step1 kernel into
chunks that process subsets of B's tile columns.

**Expected impact:** 2-3× on Step1 for large matrices (StocF, cage14),
but Step1 is already small (1-22ms).

## Optimization 6: Speculative Intersection Buffer (Low Impact)

**Problem:** When `matchedcnt > MAX_MATCHED` (32), the Step3 kernel falls back
to merge-based intersection (lines 100-108), which requires global memory reads
for tile_colidxA and csc_tile_rowidxB on every tile pair.

**Approach:** Increase MAX_MATCHED from 32 to 64 or 128. This uses more shared
memory (currently 32 × 4 bytes × 2 arrays = 256 bytes per warp, ×8 warps = 2KB)
but reduces the fallback path frequency.

With MAX_MATCHED=64: 64 × 4 × 2 × 8 = 4KB shared memory — still fits.

**Expected impact:** Small — only affects tiles with >32 matched pairs.

## Optimization 7: CPU Preprocessing Parallelism (Medium Impact)

**Problem:** CPU tile conversion is serial in GPU_SPGEMM (unlike TileSpGEMM
which uses OpenMP). For af_shell10 (1.5M rows), CPU preprocessing takes >300s.

**Approach:** Add OpenMP parallelism to `csr2tile()` and `build_csc_tiles()`
in `tile_format.h`. The per-tile-row loop in csr2tile Step 1-3 is embarrassingly
parallel.

**Expected impact:** 4-8× on preprocessing (depending on core count), enabling
completion of the 25 matrices that currently timeout.

## Priority Order

| Priority | Optimization | Expected Impact | Effort |
|---|---|---|---|
| **1** | Two-pass accumulation (Opt 1) | 1.5-2× on Step3 | Medium |
| **2** | Fuse/pre-alloc CUB scans (Opt 3) | 10-30% on small matrices | Low |
| **3** | CPU preprocessing OpenMP (Opt 7) | 4-8× on preprocessing | Low |
| **4** | Step1 shared memory (Opt 5) | 2-3× on Step1 for large | Medium |
| **5** | Warp-level scheduling (Opt 4) | 10-20% load balance | Medium |
| **6** | Shared memory accumulator (Opt 2) | Alternative to Opt 1 | Medium |
| **7** | Increase MAX_MATCHED (Opt 6) | Small | Low |

## Verification Plan

After each optimization:
1. Compare nnzC against cuSPARSE on the 8-matrix test set
2. Run the 5-matrix NSight profile to verify occupancy improvement
3. Run full 222-matrix benchmark and compare against baseline
