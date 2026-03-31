// Standalone cuSPARSE SpGEMM benchmark — C = A * B (A^2 when B=A)
// Uses generic cuSPARSE API (CUDA 11+), double precision
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { printf("cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, err); exit(1); } \
} while(0)

// ---- MatrixMarket loader (COO -> CSR, fp64, symmetric expansion) ----
struct CSR {
    int m, n, nnz;
    std::vector<int> row_ptr, col_idx;
    std::vector<double> val;
};

CSR load_mtx(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }

    char line[1024];
    int isSymmetric = 0, isPattern = 0;
    // Read banner
    if (!fgets(line, sizeof(line), f)) { printf("Empty file\n"); exit(1); }
    if (strstr(line, "symmetric")) isSymmetric = 1;
    if (strstr(line, "pattern")) isPattern = 1;

    // Skip comments
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    int m, n, nnz_file;
    sscanf(line, "%d %d %d", &m, &n, &nnz_file);

    // Read entries
    std::vector<int> rows, cols;
    std::vector<double> vals;
    rows.reserve(isSymmetric ? nnz_file * 2 : nnz_file);
    cols.reserve(isSymmetric ? nnz_file * 2 : nnz_file);
    vals.reserve(isSymmetric ? nnz_file * 2 : nnz_file);

    for (int i = 0; i < nnz_file; i++) {
        int r, c; double v = 1.0;
        if (isPattern) fscanf(f, "%d %d", &r, &c);
        else fscanf(f, "%d %d %lf", &r, &c, &v);
        r--; c--;  // 1-based to 0-based
        rows.push_back(r); cols.push_back(c); vals.push_back(v);
        if (isSymmetric && r != c) {
            rows.push_back(c); cols.push_back(r); vals.push_back(v);
        }
    }
    fclose(f);

    int nnz = (int)rows.size();

    // COO -> CSR
    CSR csr;
    csr.m = m; csr.n = n; csr.nnz = nnz;
    csr.row_ptr.resize(m + 1, 0);
    csr.col_idx.resize(nnz);
    csr.val.resize(nnz);

    for (int i = 0; i < nnz; i++) csr.row_ptr[rows[i] + 1]++;
    for (int i = 0; i < m; i++) csr.row_ptr[i + 1] += csr.row_ptr[i];

    std::vector<int> offset(csr.row_ptr.begin(), csr.row_ptr.end());
    for (int i = 0; i < nnz; i++) {
        int r = rows[i];
        int p = offset[r]++;
        csr.col_idx[p] = cols[i];
        csr.val[p] = vals[i];
    }

    // Sort columns within each row and deduplicate
    for (int i = 0; i < m; i++) {
        int start = csr.row_ptr[i], end = csr.row_ptr[i + 1];
        // Insertion sort
        for (int j = start + 1; j < end; j++) {
            int key_c = csr.col_idx[j]; double key_v = csr.val[j];
            int k = j - 1;
            while (k >= start && csr.col_idx[k] > key_c) {
                csr.col_idx[k + 1] = csr.col_idx[k];
                csr.val[k + 1] = csr.val[k];
                k--;
            }
            csr.col_idx[k + 1] = key_c; csr.val[k + 1] = key_v;
        }
        // Deduplicate: merge entries with same column index
        int write = start;
        for (int j = start; j < end; j++) {
            if (write > start && csr.col_idx[write - 1] == csr.col_idx[j]) {
                csr.val[write - 1] += csr.val[j];  // accumulate
            } else {
                csr.col_idx[write] = csr.col_idx[j];
                csr.val[write] = csr.val[j];
                write++;
            }
        }
        csr.row_ptr[i + 1] = write;  // will be fixed in recompact below
    }
    // Recompact row_ptr and arrays after dedup
    int new_nnz = 0;
    for (int i = 0; i < m; i++) {
        int old_start = (i == 0) ? 0 : csr.row_ptr[i];
        int count = csr.row_ptr[i + 1] - old_start;
        // Already compacted in-place above
        new_nnz += count;
    }
    // Rebuild row_ptr as prefix sum
    {
        std::vector<int> new_rp(m + 1, 0);
        // Count per row from the deduped end markers
        // Actually the dedup already set row_ptr[i+1] = write position
        // But the write positions are absolute, need to rebase
        // Simpler: recount
        std::vector<int> row_count(m, 0);
        // The current state: for row i, entries are at positions row_ptr[i]..row_ptr[i+1]-1 (deduped)
        // But row_ptr values got overwritten. Let me redo this properly.
    }
    // Actually the in-place dedup above already works correctly because:
    // - For row i, entries were at [start, end) before dedup
    // - After dedup, entries are at [start, write) and row_ptr[i+1] = write
    // - But row_ptr[i] is still the original start, so next row starts at the old position
    // This is fine as long as we recompact the arrays.
    // Let me just rebuild properly:
    {
        std::vector<int> new_col;
        std::vector<double> new_val;
        std::vector<int> new_rp(m + 1, 0);
        // row_ptr[i] = original start, row_ptr[i+1] = deduped end
        for (int i = 0; i < m; i++) {
            int start = csr.row_ptr[i];
            int end = csr.row_ptr[i + 1];
            new_rp[i + 1] = new_rp[i] + (end - start);
            for (int j = start; j < end; j++) {
                new_col.push_back(csr.col_idx[j]);
                new_val.push_back(csr.val[j]);
            }
        }
        csr.row_ptr = new_rp;
        csr.col_idx = new_col;
        csr.val = new_val;
        csr.nnz = (int)new_col.size();
    }

    return csr;
}

// ---- cuSPARSE SpGEMM: C = A * B ----
struct SpGEMMResult {
    int nnzC;
    double time_ms;
};

SpGEMMResult cusparse_spgemm(
    int m, int n, int k, int nnzA, int nnzB,
    int* d_rowPtrA, int* d_colIdxA, double* d_valA,
    int* d_rowPtrB, int* d_colIdxB, double* d_valB,
    int** d_rowPtrC_out, int** d_colIdxC_out, double** d_valC_out)
{
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    double alpha = 1.0, beta = 0.0;
    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, k, nnzA,
        d_rowPtrA, d_colIdxA, d_valA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, k, n, nnzB,
        d_rowPtrB, d_colIdxB, d_valB,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, m, n, 0,
        NULL, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Work estimation
    size_t bufSize1 = 0;
    void* dBuf1 = NULL;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufSize1, NULL));
    CHECK_CUDA(cudaMalloc(&dBuf1, bufSize1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufSize1, dBuf1));

    // Compute
    size_t bufSize2 = 0;
    void* dBuf2 = NULL;
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufSize2, NULL));
    CHECK_CUDA(cudaMalloc(&dBuf2, bufSize2));

    // Timed compute
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufSize2, dBuf2));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    // Get result
    int64_t Cm, Cn, Cnnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &Cm, &Cn, &Cnnz));

    CHECK_CUDA(cudaMalloc(d_rowPtrC_out, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(d_colIdxC_out, Cnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(d_valC_out, Cnnz * sizeof(double)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, *d_rowPtrC_out, *d_colIdxC_out, *d_valC_out));

    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // Cleanup
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    cudaFree(dBuf1); cudaFree(dBuf2);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {(int)Cnnz, (double)ms};
}

// ---- CPU reference SpGEMM for correctness check ----
CSR cpu_spgemm(const CSR& A, const CSR& B) {
    int m = A.m, n = B.n;
    CSR C;
    C.m = m; C.n = n;
    C.row_ptr.resize(m + 1, 0);

    std::vector<double> acc(n, 0.0);
    std::vector<bool> used(n, false);

    // Two-pass: count then fill
    std::vector<std::vector<int>> C_cols(m);
    std::vector<std::vector<double>> C_vals(m);

    for (int i = 0; i < m; i++) {
        std::vector<int> cols;
        for (int ja = A.row_ptr[i]; ja < A.row_ptr[i + 1]; ja++) {
            int k = A.col_idx[ja];
            double va = A.val[ja];
            for (int jb = B.row_ptr[k]; jb < B.row_ptr[k + 1]; jb++) {
                int c = B.col_idx[jb];
                if (!used[c]) { used[c] = true; cols.push_back(c); }
                acc[c] += va * B.val[jb];
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            C_cols[i].push_back(c);
            C_vals[i].push_back(acc[c]);
            acc[c] = 0.0; used[c] = false;
        }
        C.row_ptr[i + 1] = C.row_ptr[i] + (int)cols.size();
    }

    C.nnz = C.row_ptr[m];
    C.col_idx.resize(C.nnz);
    C.val.resize(C.nnz);
    for (int i = 0; i < m; i++) {
        int base = C.row_ptr[i];
        for (int j = 0; j < (int)C_cols[i].size(); j++) {
            C.col_idx[base + j] = C_cols[i][j];
            C.val[base + j] = C_vals[i][j];
        }
    }
    return C;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [-d device] [-check]\n", argv[0]);
        return 1;
    }

    const char* filename = NULL;
    int device = 0;
    bool check = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) device = atoi(argv[++i]);
        else if (strcmp(argv[i], "-check") == 0) check = true;
        else filename = argv[i];
    }
    if (!filename) { printf("No matrix file specified\n"); return 1; }

    cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("Device [%d] %s\n", device, prop.name);

    printf("Loading %s\n", filename);
    CSR A = load_mtx(filename);
    printf("A: %d x %d, nnz = %d\n", A.m, A.n, A.nnz);

    if (A.m != A.n) { printf("A must be square for A^2\n"); return 1; }

    // Compute flops
    long long flops = 0;
    for (int i = 0; i < A.m; i++)
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++)
            flops += A.row_ptr[A.col_idx[j] + 1] - A.row_ptr[A.col_idx[j]];
    printf("SpGEMM flops = %lld\n", flops);

    // Upload A to GPU
    int *d_rowPtrA, *d_colIdxA; double *d_valA;
    CHECK_CUDA(cudaMalloc(&d_rowPtrA, (A.m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colIdxA, A.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_valA, A.nnz * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_rowPtrA, A.row_ptr.data(), (A.m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdxA, A.col_idx.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_valA, A.val.data(), A.nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Check free memory
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t needed = (size_t)(A.m + 1 + A.nnz) * sizeof(int) + (size_t)A.nnz * sizeof(double);
    needed *= 2;  // A and B (same matrix)
    needed += needed;  // rough estimate for cuSPARSE buffers
    if (needed > freeMem * 0.8) {
        printf("WARNING: may OOM (need ~%zu MB, free %zu MB)\n", needed/1024/1024, freeMem/1024/1024);
    }

    // C = A * A
    int *d_rowPtrC, *d_colIdxC; double *d_valC;
    SpGEMMResult result = cusparse_spgemm(
        A.m, A.n, A.n, A.nnz, A.nnz,
        d_rowPtrA, d_colIdxA, d_valA,
        d_rowPtrA, d_colIdxA, d_valA,
        &d_rowPtrC, &d_colIdxC, &d_valC);

    printf("nnzC = %d\n", result.nnzC);
    printf("cuSPARSE time = %.2f ms, GFlops = %.2f\n",
           result.time_ms, 2.0 * flops / (result.time_ms * 1e6));

    // Correctness check
    if (check) {
        printf("\n=== Correctness Check ===\n");
        CSR C_ref = cpu_spgemm(A, A);
        printf("CPU reference: nnzC = %d\n", C_ref.nnz);

        // Download GPU result
        std::vector<int> h_rowPtrC(A.m + 1), h_colIdxC(result.nnzC);
        std::vector<double> h_valC(result.nnzC);
        cudaMemcpy(h_rowPtrC.data(), d_rowPtrC, (A.m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colIdxC.data(), d_colIdxC, result.nnzC * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_valC.data(), d_valC, result.nnzC * sizeof(double), cudaMemcpyDeviceToHost);

        if (result.nnzC != C_ref.nnz) {
            printf("FAIL: nnz mismatch: cuSPARSE=%d, CPU=%d\n", result.nnzC, C_ref.nnz);
        } else {
            int errs = 0;
            double max_reldiff = 0;
            for (int i = 0; i < A.m; i++) {
                if (h_rowPtrC[i] != C_ref.row_ptr[i]) { errs++; continue; }
                for (int j = h_rowPtrC[i]; j < h_rowPtrC[i + 1]; j++) {
                    if (h_colIdxC[j] != C_ref.col_idx[j]) { errs++; continue; }
                    double ref = C_ref.val[j], got = h_valC[j];
                    double reldiff = fabs(ref - got) / (fabs(ref) + 1e-15);
                    if (reldiff > max_reldiff) max_reldiff = reldiff;
                    if (reldiff > 1e-6) errs++;
                }
            }
            if (errs == 0)
                printf("PASS (max relative diff = %.2e)\n", max_reldiff);
            else
                printf("FAIL: %d errors (max relative diff = %.2e)\n", errs, max_reldiff);
        }
    }

    cudaFree(d_rowPtrA); cudaFree(d_colIdxA); cudaFree(d_valA);
    cudaFree(d_rowPtrC); cudaFree(d_colIdxC); cudaFree(d_valC);
    return 0;
}
