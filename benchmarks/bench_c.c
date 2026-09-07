#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include "../src/nanogemm_kernel.h"

int main() {
    int dims[] = {16, 32, 64, 128, 256, 512};
    int iters[] = {200000, 100000, 20000, 5000, 1000, 100};
    int num_sizes = 6;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    printf("\n=== Raw C Microkernel Performance ===\n");
    printf("%-12s | %-15s | %-15s\n", "Matrix Size", "Latency (us)", "GFLOPS");
    printf("--------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int dim = dims[s];
        int iter = iters[s];
        int N = dim * dim;

        float* A = (float*)_aligned_malloc(N * sizeof(float), 32);
        float* B = (float*)_aligned_malloc(N * sizeof(float), 32);
        float* C = (float*)_aligned_malloc(N * sizeof(float), 32);

        for (int i = 0; i < N; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
            C[i] = 0.0f;
        }

        // Warmup
        nanogemm_matmul(dim, dim, dim, A, B, C);

        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        for (int i = 0; i < iter; i++) {
            nanogemm_matmul(dim, dim, dim, A, B, C);
        }
        QueryPerformanceCounter(&t1);

        double elapsed_sec = (double)(t1.QuadPart - t0.QuadPart) / (freq.QuadPart * iter);
        double latency_us = elapsed_sec * 1e6;
        double gflops = (2.0 * dim * dim * dim / elapsed_sec) / 1e9;

        printf("%-12s | %12.3f us | %12.2f GFLOPS\n", 
               dims[s] == 16 ? "16x16" : dims[s] == 32 ? "32x32" : dims[s] == 64 ? "64x64" : dims[s] == 128 ? "128x128" : dims[s] == 256 ? "256x256" : "512x512",
               latency_us, gflops);

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C);
    }
    printf("--------------------------------------------------\n\n");
    return 0;
}
