/*
 * SOR-like loop kernels for benchmarking Python vs Numba vs C.
 *
 * Provides a 1D (single for-loop) and 2D (nested for-loop) kernel
 * that mirror the SOR update pattern used in the diffusion solver.
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o sor_kernels.so sor_kernels.c
 */

/* 1D SOR-like kernel: single for-loop over a 1D array.
 *
 *   y[i] = omega * 0.5 * (y[i-1] + y[i+1]) + (1 - omega) * y[i]
 *
 * for i in [1, n-2].  Boundary values y[0] and y[n-1] are untouched.
 */
void sor_1d_kernel(double *y, int n, double omega) {
    for (int i = 1; i < n - 1; i++) {
        y[i] = omega * 0.5 * (y[i - 1] + y[i + 1])
             + (1.0 - omega) * y[i];
    }
}

/* 2D SOR-like kernel: nested for-loop over a 2D array (row-major).
 *
 *   y[i,j] = omega * 0.25 * (y[i+1,j] + y[i-1,j] + y[i,j+1] + y[i,j-1])
 *          + (1 - omega) * y[i,j]
 *
 * for j in [1, ncols-2], i in [1, nrows-2].
 * Boundary rows/columns are untouched (no periodic BC, for simplicity).
 */
void sor_2d_kernel(double *y, int nrows, int ncols, double omega) {
    for (int j = 1; j < ncols - 1; j++) {
        for (int i = 1; i < nrows - 1; i++) {
            int idx   = i * ncols + j;
            int right = (i + 1) * ncols + j;
            int left  = (i - 1) * ncols + j;
            int up    = i * ncols + (j + 1);
            int down  = i * ncols + (j - 1);
            y[idx] = omega * 0.25 * (y[right] + y[left] + y[up] + y[down])
                   + (1.0 - omega) * y[idx];
        }
    }
}
