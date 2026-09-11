#ifndef NANOGEMM_KERNEL_H
#define NANOGEMM_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef BUILDING_NANOGEMM
    #define NANOGEMM_API __declspec(dllexport)
  #else
    #define NANOGEMM_API __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define NANOGEMM_API __attribute__((visibility("default")))
  #else
    #define NANOGEMM_API
  #endif
#endif

NANOGEMM_API void nanogemm_sgemm(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc
);

NANOGEMM_API void nanogemm_matmul(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C
);

NANOGEMM_API void nanogemm_bmm(
    int batch_count,
    int M, int N, int K,
    const float* A, int stride_a,
    const float* B, int stride_b,
    float* C, int stride_c
);

NANOGEMM_API void nanogemm_gemm_i8i8i32(
    int M, int N, int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc
);

NANOGEMM_API const char* nanogemm_simd_isa(void);

#ifdef __cplusplus
}
#endif

#endif /* NANOGEMM_KERNEL_H */
