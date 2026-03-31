# NSight Compute Profiling Analysis: GPU_SPGEMM on A100

## Profiled Matrices

| Matrix | n | avg nnz/row | Category | GPU time |
|---|---|---|---|---|
| rgg_n_2_21_s0 | 1,652,680 | 21.0 | Large sparse (extreme speedup) | 0.84 ms |
| Queen_4147 | 16,830 | 170.3 | Small dense (large speedup) | 9.69 ms |
| consph | 217,918 | 53.4 | Medium FEM (typical) | 11.81 ms |
| StocF-1465 | 108,384 | 93.8 | Dense FEM (TileSpGEMM close) | 15.06 ms |
| cage14 | 49,152 | 39.0 | Small sparse (TileSpGEMM close) | 3.69 ms |

Note: rgg_n_2_21_s0 and Queen_4147 had too few tiles for the custom kernels
to register in profiling (Step1/Step2 used CUB scan kernels only). The analysis
focuses on consph, StocF-1465, and cage14 where all custom kernels are captured.

## Kernel-Level Analysis

### step3_numeric_kernel_v2 (Dominant Kernel)

This is the numeric SpGEMM kernel — the most important for performance.

| Metric | consph | StocF-1465 | cage14 |
|---|---|---|---|
| **Duration** | 14.53 µs | 72.97 µs | 165.43 µs |
| **Occupancy** | 20.91% | 20.80% | 23.50% |
| **Registers/thread** | 76 | 76 | 76 |
| **Shared memory** | 2.05 KB static | 2.05 KB static | 2.05 KB static |
| **Memory throughput** | 29.02% | 40.95% | 48.47% |
| **L1 throughput** | 12.81% | 12.67% | 14.36% |
| **L2 throughput** | 7.06% | 6.28% | 8.25% |
| **DRAM throughput** | 1.42% | 2.01% | 2.38% |
| **Waves/SM** | 250 | 6,981 | 15,067 |

**Key observations:**

1. **Low occupancy (~21-24%):** The kernel uses 76 registers per thread. On A100
   (sm_80), each SM has 65,536 registers. With 256 threads per block (8 warps),
   each block needs 76 × 256 = 19,456 registers. Max 3 blocks per SM =
   3 × 8 = 24 warps out of 64 max = 37.5% theoretical occupancy. The achieved
   ~21% is below theoretical, suggesting launch configuration or shared memory
   limits further reduce occupancy.

2. **Memory-bound but underutilizing bandwidth:** Memory throughput is 29-48% —
   far from saturated. DRAM throughput is only 1.4-2.4%, meaning almost all
   data comes from L1/L2 cache. This is expected for tile-based SpGEMM where
   16×16 tiles fit in cache. The kernel is **latency-bound**, not bandwidth-bound.

3. **Very high waves/SM on larger matrices:** cage14 has 15,067 waves/SM,
   meaning the SM processes thousands of tile-pairs sequentially. Each wave
   has low occupancy, so the SM spends significant time on scheduling overhead.

4. **L1 throughput is modest (12-14%):** The kernel reads tile data through
   `__ldg()` (read-only cache path). The low L1 throughput combined with low
   DRAM throughput suggests the kernel is **compute-latency-bound** — waiting
   for FP64 multiply-accumulate operations rather than memory.

### step2_symbolic_kernel

| Metric | consph | StocF-1465 | cage14 |
|---|---|---|---|
| **Duration** | 1.69 µs | 27.08 µs | 66.96 µs |
| **Occupancy** | 59.41% | 64.38% | 63.91% |
| **Registers/thread** | 40 | 40 | 40 |
| **Memory throughput** | 38.52% | 61.88% | 62.68% |
| **L1 throughput** | 47.91% | 39.52% | 34.78% |
| **DRAM throughput** | 1.89% | 3.03% | 3.07% |
| **Waves/SM** | 7.82 | 218 | 471 |

**Key observations:**

1. **Higher occupancy than Step3 (59-64%):** Fewer registers (40 vs 76) allows
   more concurrent warps. This kernel does symbolic intersection — no FP64 math,
   just integer comparisons and bitmask operations.

2. **L1-cache dominated:** High L1 throughput (35-48%) with minimal DRAM access.
   The tile column indices fit well in L1 cache for sorted-merge intersection.

3. **Scales linearly with tiles:** Duration scales from 1.7µs (consph, 458K tiles)
   to 67µs (cage14, 263K tiles but more intersections per tile).

### step1_count_kernel and step1_fill_kernel

| Metric | consph count | consph fill | StocF count | StocF fill |
|---|---|---|---|---|
| **Duration** | 37.31 µs | 85.92 µs | 1.43 µs | 18.99 µs |
| **Occupancy** | 53.85% | 54.71% | 15.62% | 18.29% |
| **Memory throughput** | 31.25% | 25.63% | 13.58% | 7.80% |
| **L1 throughput** | 58.41% | 45.31% | 27.60% | 14.31% |
| **Shared memory** | 0/2.61 KB | 0/2.61 KB | 0/45.79 KB | 0/45.79 KB |

**Key observations:**

1. **Shared memory scales with tile columns:** consph (13K tile columns) uses
   2.61 KB dynamic shared memory for the bitmask, while StocF (6.7K tile columns)
   uses 45.79 KB. Wait — StocF has fewer tile columns but more shared memory?
   This is because `nmasks = (tilenB + 31) / 32` words per warp, and with more
   warps per block, total shared memory grows. The high shared memory on StocF
   limits occupancy to 15-18%.

2. **Step1 is fast for both:** Even the "fill" pass (85µs for consph) is small
   compared to Step3 (14ms). Step1 is not the bottleneck for these matrices.

### step2_build_ptrRowC_kernel

| Metric | consph | StocF-1465 | cage14 |
|---|---|---|---|
| **Duration** | 93.73 µs | 2.40 µs | 5.18 µs |
| **Occupancy** | 84.06% | 93.79% | 92.14% |
| **Memory throughput** | 248.81% | 357.79% | 360.11% |
| **L2 throughput** | 72.18% | 78.91% | 78.65% |
| **DRAM throughput** | 12.21% | 17.55% | 17.66% |

This kernel builds the CSR row pointers for C from the symbolic result. Very high
occupancy (84-94%) and memory throughput (249-360%) — it's a simple prefix-sum-like
operation that saturates the memory subsystem. Duration is small.

### CUB DeviceScan Kernels

The CUB exclusive scan kernels are called for prefix sums. They have moderate
occupancy (24-30%) due to high register usage (65 regs) but are highly optimized.
For StocF and cage14, the scan kernel takes 128-272µs — significant compared to
the custom kernels.

## Why GPU_SPGEMM Is Slower on Some Matrices

For **StocF-1465** and **cage14** (where GPU_SPGEMM's advantage is smallest):

1. **High register pressure in Step3:** 76 registers/thread limits occupancy to
   ~21%. With FP64, each register holds one double — the 16 accumulator registers
   (`r0..r15`) alone use 16 registers. The remaining 60 registers hold loop
   variables, tile pointers, and intermediate values.

2. **CUB scan overhead becomes proportionally larger:** For matrices where Step3
   is dominant (cage14: 165µs Step3 + 272µs CUB scan), the CUB infrastructure
   overhead is 62% of total time. TileSpGEMM uses thrust for scans which may have
   different performance characteristics.

3. **Moderate tile density:** Both StocF and cage14 have moderate compression
   ratios (27.0 and 6.9). With more nonzeros per tile, the Step3 kernel does
   more useful work per tile-pair, but the low occupancy limits throughput.

## Optimization Opportunities

1. **Reduce register pressure in Step3:** The 16 accumulator registers for the
   dense 16×16 output tile could be reduced by processing fewer columns per pass,
   trading registers for loop iterations. Target: 48 registers → 50% occupancy.

2. **Fuse CUB scans:** The separate DeviceScanInit + DeviceScan calls add kernel
   launch overhead. For small arrays, a single-block scan would be faster.

3. **Improve Step1 shared memory usage:** For matrices with many tile columns
   (large `nmasks`), the dynamic shared memory limits occupancy. Consider using
   global memory for the bitmask when nmasks > threshold.

4. **Warp specialization in Step3:** With only 21% occupancy, many SM resources
   are idle. A warp-specialized approach where some warps prefetch tile data while
   others compute could improve latency hiding.
