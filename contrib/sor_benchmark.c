/*
 * Native C benchmark for SOR kernels — no ctypes overhead.
 *
 * Usage:
 *   ./sor_benchmark <n_1d> <repeats_1d> <n_2d> <repeats_2d> <omega>
 *
 * Output (machine-parseable, one line per result):
 *   1d <total_seconds>
 *   2d <total_seconds>
 *
 * Compile & link:
 *   gcc -O2 -o sor_benchmark sor_benchmark.c -L. -lsor_kernels -Wl,-rpath,'$ORIGIN'
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Declarations for functions in sor_kernels.so */
extern void sor_1d_kernel(double *y, int n, double omega);
extern void sor_2d_kernel(double *y, int nrows, int ncols, double omega);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Fill array with linspace(0, 1, n) */
static void fill_linspace(double *y, int n) {
    for (int i = 0; i < n; i++)
        y[i] = (double)i / (n - 1);
}

/* Fill 2D array (nrows x ncols, row-major) with zeros, last column = 1 */
static void fill_2d(double *y, int nrows, int ncols) {
    for (int i = 0; i < nrows * ncols; i++)
        y[i] = 0.0;
    for (int i = 0; i < nrows; i++)
        y[i * ncols + (ncols - 1)] = 1.0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <n_1d> <repeats_1d> <n_2d> <repeats_2d> <omega>\n", argv[0]);
        return 1;
    }

    int n_1d      = atoi(argv[1]);
    int repeats_1d = atoi(argv[2]);
    int n_2d      = atoi(argv[3]);
    int repeats_2d = atoi(argv[4]);
    double omega  = atof(argv[5]);

    int nrows = n_2d + 1;
    int ncols = n_2d + 1;

    /* Allocate arrays */
    double *y1d = (double *)malloc(n_1d * sizeof(double));
    double *y2d = (double *)malloc(nrows * ncols * sizeof(double));

    /* --- 1D benchmark --- */
    double t0 = now_sec();
    for (int r = 0; r < repeats_1d; r++) {
        fill_linspace(y1d, n_1d);
        sor_1d_kernel(y1d, n_1d, omega);
    }
    double t1d = now_sec() - t0;

    /* --- 2D benchmark --- */
    t0 = now_sec();
    for (int r = 0; r < repeats_2d; r++) {
        fill_2d(y2d, nrows, ncols);
        sor_2d_kernel(y2d, nrows, ncols, omega);
    }
    double t2d = now_sec() - t0;

    printf("1d %.9f\n", t1d);
    printf("2d %.9f\n", t2d);

    free(y1d);
    free(y2d);
    return 0;
}
