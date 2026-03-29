# TileSpGEMM GPU Hang Bug: Missing `__syncwarp()` on Volta+ GPUs

## Summary

TileSpGEMM hangs indefinitely on NVIDIA Volta and later GPUs (V100, A100, etc.)
when processing sparse matrices with large tile dimensions (>16384 tile columns).
The root cause is missing `__syncwarp()` synchronization in the nsparse hash-based
SpGEMM kernels, which rely on implicit warp-synchronous execution that is no longer
guaranteed on Volta+ architectures.

## Affected Code

- **File:** `TileSpGEMM/src/spgemm_nsparse_kernel.h`
- **Kernel:** `calculate_value_col_bin_pwarp` (line ~696)
- **Also affected:** `set_row_nz_bin_pwarp` (line ~331)

## Trigger Condition

The bug triggers when `blknB > NUMCOLC_SPA_OR_HASH_TH` (i.e., tile column count
exceeds 16384), which routes SpGEMM through the nsparse bin-based code path instead
of the SPA-based path. This happens for matrices with many rows/columns relative to
the tile size (BLOCK_SIZE=16):

- **mc2depi:** 525,825 rows -> 32,865 tile rows -> hangs
- **webbase-1M:** 1,000,005 rows -> 62,501 tile rows -> hangs
- **af_shell10:** 1,508,065 rows -> 94,255 tile rows -> hangs

Matrices with fewer than ~262,000 rows (16,384 tiles) take the SPA path and work fine.

## Root Cause

### Background: Warp Execution Model Change

Before NVIDIA Volta (compute capability < 7.0), all 32 threads in a warp executed
in strict lockstep (SIMT). Code could rely on implicit synchronization: if thread 0
wrote to shared memory, thread 1 could immediately read it because they executed the
same instruction simultaneously.

Starting with Volta (sm_70), NVIDIA introduced **Independent Thread Scheduling**.
Threads within a warp can now diverge and execute at different rates. Shared memory
writes by one thread are no longer guaranteed to be visible to other threads without
explicit synchronization via `__syncwarp()`.

### The Bug

The `calculate_value_col_bin_pwarp` kernel processes rows in groups of `PWARP=4`
threads. Each group manages a 16-slot hash table (`B_PWMIN=16`) in shared memory.
The kernel has three phases:

**Phase 1 — Hash Insert (line 745-775):**
Each thread iterates over its portion of intermediate products and inserts column
indices into the shared-memory hash table using `atomicCAS`.

**Phase 2 — Compaction (line 777-784):**
Threads compact the hash table entries into positions 0..nz-1 using `atomicAdd` on
a global counter `d_nz[rid]`.

**Phase 3 — Sorting and Output (line 788-801):**
Threads sort the compacted entries by rank and write to global memory.

The critical bug is at **line 735-738**:

```cuda
if (tid == 0) {
    d_nz[rid] = 0;    // Only thread 0 resets the counter
}
// NO __syncwarp() here!
```

`d_nz` is `bin->d_row_nz`, which was previously populated by `set_row_nnz` with
the actual nnz count per row (e.g., 15 for mc2depi). Thread 0 resets it to 0 before
the hash insert phase. But without `__syncwarp()`, **threads 1-3 can race ahead to
the compaction phase (line 779) before thread 0's write is visible.** They call:

```cuda
index = atomicAdd(d_nz + rid, 1);                    // returns 15 (stale value!)
shared_check[soffset + index] = shared_check[soffset + j];  // writes at soffset+15
```

Since `d_nz[rid]` still holds the stale value (e.g., 15), `atomicAdd` returns 15,
16, 17, etc. The compaction writes to `shared_check[soffset + 15]`, which is the
**last slot** of this row's 16-slot region. Index 16 writes to `soffset + 16`, which
is the **first slot of the adjacent row's hash table** in shared memory.

### The Cascade

This corruption propagates:

1. Row A's compaction overflows into Row B's 16-slot region in shared memory
2. Row B's hash table now contains garbage values (column indices from Row A)
3. When Row B's threads try to insert their keys, the `while(1)` probing loop
   finds no empty slot (`-1`) and no matching key in the corrupted table
4. The loop probes all 16 slots repeatedly, finding only Row A's stale values
5. **Infinite loop** — the kernel never terminates

Because the shared memory layout packs 64 rows per block (256 threads / 4 PWARP),
a single corrupted row can cascade to corrupt its neighbor, which corrupts the next,
etc. With 32,865 rows across 514 thread blocks on an A100 (108 SMs), the corruption
affects enough warps to effectively hang the entire GPU.

## Why It Worked on Pre-Volta GPUs

The TileSpGEMM paper (PPoPP '22) was evaluated on RTX 3060 and RTX 3090, both
Ampere (sm_86). However, the nsparse library it incorporates was originally written
for older GPU architectures. The commented-out `__syncwarp()` calls throughout the
code (visible as `//__syncwarp();`) suggest the authors were aware of the need but
disabled them, likely because:

1. The code was originally developed for Pascal (sm_61) where warp-sync was implicit
2. On Ampere, the bug only manifests for large-tile-dimension matrices (>16K tiles)
3. The paper's 142-matrix benchmark may not have included matrices that trigger this
   specific code path, or the hang was misattributed to preprocessing overhead

## Fix

Uncomment/add `__syncwarp()` at three points in `calculate_value_col_bin_pwarp`:

```cuda
// After d_nz reset (before hash insert phase)
if (tid == 0) {
    d_nz[rid] = 0;
}
__syncwarp();  // Ensure all threads see d_nz[rid] = 0

// After hash insert phase (before compaction)
// ... hash insert loop ...
__syncwarp();  // Ensure all inserts are visible before compaction

// After compaction (before sorting/output)
// ... compaction loop ...
__syncwarp();  // Ensure compacted values are visible before sorting
```

The same fix should be applied to `set_row_nz_bin_pwarp` (add `__syncwarp()` after
the shared memory initialization loop, before the early return check).

## Verification

After the fix, all three previously-hanging matrices complete:

| Matrix | Rows | Tile Rows | Before Fix | After Fix |
|--------|------|-----------|------------|-----------|
| mc2depi | 525,825 | 32,865 | Infinite hang | 13.39 ms |
| webbase-1M | 1,000,005 | 62,501 | Infinite hang | 33.34 ms |
| af_shell10 | 1,508,065 | 94,255 | Infinite hang | 73.03 ms |

## Related Issues

The same missing-syncwarp pattern exists in other nsparse kernels in the file
(`set_row_nz_bin_each`, `set_row_nz_bin_each_tb`, `calculate_value_col_bin_each`,
`calculate_value_col_bin_each_tb`, etc.) but these are only triggered for rows
assigned to higher bins (intermediate products > 32). A comprehensive audit of all
`while(1)` hash probing loops in `spgemm_nsparse_kernel.h` is recommended.
