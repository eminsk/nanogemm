#ifndef BUILDING_NANOGEMM
#define BUILDING_NANOGEMM
#endif
#include "nanogemm_kernel.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define NANOGEMM_ARM_NEON 1
#elif defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define NANOGEMM_X86_AVX2 1
#else
    #define NANOGEMM_SCALAR 1
#endif


#define BLOCK_M 64
#define BLOCK_N 128
#define BLOCK_K 128

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#if defined(NANOGEMM_ARM_NEON)
/* -------------------------------------------------------------
 * ARM NEON 4x8 Microkernel (Apple Silicon M1/M2/M3/M4 & aarch64)
 * Computes a 4x8 tile of C from 4xK of A and Kx8 of B
 * ------------------------------------------------------------- */
static inline void sgemm_microkernel_4x8_neon(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f);
    float32x4_t c31 = vdupq_n_f32(0.0f);

    for (int k = 0; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(&B[k * ldb]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);

        float a0 = A[0 * lda + k];
        c00 = vmlaq_n_f32(c00, b0, a0);
        c01 = vmlaq_n_f32(c01, b1, a0);

        float a1 = A[1 * lda + k];
        c10 = vmlaq_n_f32(c10, b0, a1);
        c11 = vmlaq_n_f32(c11, b1, a1);

        float a2 = A[2 * lda + k];
        c20 = vmlaq_n_f32(c20, b0, a2);
        c21 = vmlaq_n_f32(c21, b1, a2);

        float a3 = A[3 * lda + k];
        c30 = vmlaq_n_f32(c30, b0, a3);
        c31 = vmlaq_n_f32(c31, b1, a3);
    }

    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta  = vdupq_n_f32(beta);

    #define STORE_ROW_NEON(row, c_lo, c_hi) \
    do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            vst1q_f32(dst,     vmulq_f32(c_lo, valpha)); \
            vst1q_f32(dst + 4, vmulq_f32(c_hi, valpha)); \
        } else { \
            float32x4_t old_lo = vld1q_f32(dst); \
            float32x4_t old_hi = vld1q_f32(dst + 4); \
            vst1q_f32(dst,     vmlaq_f32(vmulq_f32(old_lo, vbeta), c_lo, valpha)); \
            vst1q_f32(dst + 4, vmlaq_f32(vmulq_f32(old_hi, vbeta), c_hi, valpha)); \
        } \
    } while(0)

    STORE_ROW_NEON(0, c00, c01);
    STORE_ROW_NEON(1, c10, c11);
    STORE_ROW_NEON(2, c20, c21);
    STORE_ROW_NEON(3, c30, c31);
    #undef STORE_ROW_NEON
}
#endif

#if defined(NANOGEMM_X86_AVX2)
#if defined(__clang__) || defined(__GNUC__)
    #define NANOGEMM_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
    #define NANOGEMM_AVX2_TARGET
#endif

/* -------------------------------------------------------------
 * AVX2 + FMA 6x16 Microkernel
 * Computes a 6x16 tile of C from 6xK of A and Kx16 of B
 * ------------------------------------------------------------- */
NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_6x16_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW(row, r0, r1) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
            _mm256_storeu_ps(dst + 8, _mm256_mul_ps(r1, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 cur1 = _mm256_loadu_ps(dst + 8); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
            _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(r1, valpha, _mm256_mul_ps(cur1, vbeta))); \
        } \
    } while (0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);
    STORE_ROW(4, c40, c41);
    STORE_ROW(5, c50, c51);

    #undef STORE_ROW
}

/* -------------------------------------------------------------
 * AVX2 + FMA 4x16 Microkernel (Boundary for M=4,16,28...)
 * ------------------------------------------------------------- */
NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_4x16_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW_4(row, r0, r1) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
            _mm256_storeu_ps(dst + 8, _mm256_mul_ps(r1, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 cur1 = _mm256_loadu_ps(dst + 8); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
            _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(r1, valpha, _mm256_mul_ps(cur1, vbeta))); \
        } \
    } while (0)

    STORE_ROW_4(0, c00, c01);
    STORE_ROW_4(1, c10, c11);
    STORE_ROW_4(2, c20, c21);
    STORE_ROW_4(3, c30, c31);

    #undef STORE_ROW_4
}

/* -------------------------------------------------------------
 * AVX2 + FMA 2x16 Microkernel (Boundary for M=2,8,14,32...)
 * ------------------------------------------------------------- */
NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_2x16_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW_2(row, r0, r1) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
            _mm256_storeu_ps(dst + 8, _mm256_mul_ps(r1, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 cur1 = _mm256_loadu_ps(dst + 8); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
            _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(r1, valpha, _mm256_mul_ps(cur1, vbeta))); \
        } \
    } while (0)

    STORE_ROW_2(0, c00, c01);
    STORE_ROW_2(1, c10, c11);

    #undef STORE_ROW_2
}

/* -------------------------------------------------------------
 * AVX2 + FMA 6x8, 4x8, 2x8 Microkernels (For N=8 boundaries)
 * ------------------------------------------------------------- */
NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_6x8_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b0, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b0, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b0, c5);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW_6x8(row, r0) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
        } \
    } while (0)

    STORE_ROW_6x8(0, c0);
    STORE_ROW_6x8(1, c1);
    STORE_ROW_6x8(2, c2);
    STORE_ROW_6x8(3, c3);
    STORE_ROW_6x8(4, c4);
    STORE_ROW_6x8(5, c5);

    #undef STORE_ROW_6x8
}

NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_4x8_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b0, c3);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW_4x8(row, r0) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
        } \
    } while (0)

    STORE_ROW_4x8(0, c0);
    STORE_ROW_4x8(1, c1);
    STORE_ROW_4x8(2, c2);
    STORE_ROW_4x8(3, c3);

    #undef STORE_ROW_4x8
}

NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_2x8_avx2(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b0, c1);
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    #define STORE_ROW_2x8(row, r0) do { \
        float* dst = &C[(row) * ldc]; \
        if (is_first_k && beta == 0.0f) { \
            _mm256_storeu_ps(dst, _mm256_mul_ps(r0, valpha)); \
        } else { \
            __m256 cur0 = _mm256_loadu_ps(dst); \
            __m256 vbeta = _mm256_set1_ps(beta); \
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(r0, valpha, _mm256_mul_ps(cur0, vbeta))); \
        } \
    } while (0)

    STORE_ROW_2x8(0, c0);
    STORE_ROW_2x8(1, c1);

    #undef STORE_ROW_2x8
}
#endif


/* -------------------------------------------------------------
 * Fallback Edge Kernel (handles odd non-aligned remainders)
 * ------------------------------------------------------------- */
static inline void sgemm_edge_kernel(
    int m_count, int n_count, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha, float beta,
    int is_first_k)
{
    for (int i = 0; i < m_count; ++i) {
        for (int j = 0; j < n_count; ++j) {
            float sum = 0.0f;
            int k = 0;

            for (; k <= K - 4; k += 4) {
                sum += A[i * lda + k + 0] * B[(k + 0) * ldb + j];
                sum += A[i * lda + k + 1] * B[(k + 1) * ldb + j];
                sum += A[i * lda + k + 2] * B[(k + 2) * ldb + j];
                sum += A[i * lda + k + 3] * B[(k + 3) * ldb + j];
            }
            for (; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }

            if (is_first_k && beta == 0.0f) {
                C[i * ldc + j] = alpha * sum;
            } else {
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
    }
}

/* -------------------------------------------------------------
 * Cache-Blocked SGEMM Driver
 * ------------------------------------------------------------- */
#if defined(NANOGEMM_X86_AVX2)
NANOGEMM_AVX2_TARGET
#endif
NANOGEMM_API void nanogemm_sgemm(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    for (int kb = 0; kb < K; kb += BLOCK_K) {
        int k_block = MIN(BLOCK_K, K - kb);
        int is_first_k = (kb == 0);
        float current_beta = is_first_k ? beta : 1.0f;

        for (int mb = 0; mb < M; mb += BLOCK_M) {
            int m_block = MIN(BLOCK_M, M - mb);

            for (int nb = 0; nb < N; nb += BLOCK_N) {
                int n_block = MIN(BLOCK_N, N - nb);

                const float* A_tile = &A[mb * lda + kb];
                const float* B_tile = &B[kb * ldb + nb];
                float* C_tile = &C[mb * ldc + nb];

#if defined(NANOGEMM_ARM_NEON)
                int i = 0;
                for (; i <= m_block - 4; i += 4) {
                    int j = 0;
                    for (; j <= n_block - 8; j += 8) {
                        sgemm_microkernel_4x8_neon(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    if (j < n_block) {
                        sgemm_edge_kernel(
                            4, n_block - j, k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                }
                if (i < m_block) {
                    sgemm_edge_kernel(
                        m_block - i, n_block, k_block,
                        &A_tile[i * lda], lda,
                        &B_tile[0], ldb,
                        &C_tile[i * ldc], ldc,
                        alpha, current_beta, is_first_k);
                }
#elif defined(NANOGEMM_X86_AVX2)
                int i = 0;
                // 1. Process 6-row blocks
                for (; i <= m_block - 6; i += 6) {
                    int j = 0;
                    for (; j <= n_block - 16; j += 16) {
                        sgemm_microkernel_6x16_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    for (; j <= n_block - 8; j += 8) {
                        sgemm_microkernel_6x8_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    if (j < n_block) {
                        sgemm_edge_kernel(
                            6, n_block - j, k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                }
                // 2. Process 4-row boundary blocks (e.g. 16 = 6 + 6 + 4)
                for (; i <= m_block - 4; i += 4) {
                    int j = 0;
                    for (; j <= n_block - 16; j += 16) {
                        sgemm_microkernel_4x16_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    for (; j <= n_block - 8; j += 8) {
                        sgemm_microkernel_4x8_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    if (j < n_block) {
                        sgemm_edge_kernel(
                            4, n_block - j, k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                }
                // 3. Process 2-row boundary blocks (e.g. 32 = 6*5 + 2)
                for (; i <= m_block - 2; i += 2) {
                    int j = 0;
                    for (; j <= n_block - 16; j += 16) {
                        sgemm_microkernel_2x16_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    for (; j <= n_block - 8; j += 8) {
                        sgemm_microkernel_2x8_avx2(
                            k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                    if (j < n_block) {
                        sgemm_edge_kernel(
                            2, n_block - j, k_block,
                            &A_tile[i * lda], lda,
                            &B_tile[j], ldb,
                            &C_tile[i * ldc + j], ldc,
                            alpha, current_beta, is_first_k);
                    }
                }
                // 4. Remaining 1 row for odd dimensions
                if (i < m_block) {
                    sgemm_edge_kernel(
                        m_block - i, n_block, k_block,
                        &A_tile[i * lda], lda,
                        &B_tile[0], ldb,
                        &C_tile[i * ldc], ldc,
                        alpha, current_beta, is_first_k);
                }
#else
                sgemm_edge_kernel(
                    m_block, n_block, k_block,
                    &A_tile[0], lda,
                    &B_tile[0], ldb,
                    &C_tile[0], ldc,
                    alpha, current_beta, is_first_k);
#endif
            }
        }
    }
}

NANOGEMM_API void nanogemm_matmul(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C)
{
    nanogemm_sgemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

NANOGEMM_API const char* nanogemm_simd_isa(void)
{
#if defined(NANOGEMM_ARM_NEON)
    return "ARM NEON (128-bit SIMD, 4x8 register tiling)";
#elif defined(__AVX2__) && defined(__FMA__)
    return "AVX2+FMA (256-bit SIMD, 6x16 register tiling)";
#elif defined(__AVX2__)
    return "AVX2 (256-bit SIMD)";
#elif defined(__SSE2__)
    return "SSE2 (128-bit SIMD)";
#else
    return "Generic Scalar";
#endif
}

/* -------------------------------------------------------------
 * Batched SGEMM (BMM)
 * ------------------------------------------------------------- */
NANOGEMM_API void nanogemm_bmm(
    int batch_count,
    int M, int N, int K,
    const float* A, int stride_a,
    const float* B, int stride_b,
    float* C, int stride_c)
{
    if (batch_count <= 0 || M <= 0 || N <= 0 || K <= 0) return;

    for (int b = 0; b < batch_count; ++b) {
        const float* cur_a = A + (stride_a ? (size_t)b * stride_a : 0);
        const float* cur_b = B + (stride_b ? (size_t)b * stride_b : 0);
        float* cur_c = C + (size_t)b * stride_c;
        nanogemm_matmul(M, N, K, cur_a, cur_b, cur_c);
    }
}

/* -------------------------------------------------------------
 * INT8 Matrix Multiplication (A: int8_t, B: int8_t -> C: int32_t)
 * ------------------------------------------------------------- */
static inline void sgemm_i8_edge(
    int m_count, int n_count, int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc)
{
    for (int i = 0; i < m_count; ++i) {
        for (int j = 0; j < n_count; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

#if defined(NANOGEMM_X86_AVX2)
NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_4x16_i8_avx2(
    int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc)
{
    __m256i c00 = _mm256_setzero_si256();
    __m256i c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256();
    __m256i c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256();
    __m256i c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256();
    __m256i c31 = _mm256_setzero_si256();

    int k = 0;
    for (; k <= K - 2; k += 2) {
        __m128i b_raw0_0 = _mm_loadl_epi64((const __m128i*)&B[(k + 0) * ldb]);
        __m128i b_raw1_0 = _mm_loadl_epi64((const __m128i*)&B[(k + 0) * ldb + 8]);
        __m128i b_raw0_1 = _mm_loadl_epi64((const __m128i*)&B[(k + 1) * ldb]);
        __m128i b_raw1_1 = _mm_loadl_epi64((const __m128i*)&B[(k + 1) * ldb + 8]);

        __m256i b0_0 = _mm256_cvtepi8_epi32(b_raw0_0);
        __m256i b1_0 = _mm256_cvtepi8_epi32(b_raw1_0);
        __m256i b0_1 = _mm256_cvtepi8_epi32(b_raw0_1);
        __m256i b1_1 = _mm256_cvtepi8_epi32(b_raw1_1);

        __m256i a0_0 = _mm256_set1_epi32((int32_t)A[0 * lda + k + 0]);
        __m256i a0_1 = _mm256_set1_epi32((int32_t)A[0 * lda + k + 1]);
        c00 = _mm256_add_epi32(c00, _mm256_add_epi32(_mm256_mullo_epi32(a0_0, b0_0), _mm256_mullo_epi32(a0_1, b0_1)));
        c01 = _mm256_add_epi32(c01, _mm256_add_epi32(_mm256_mullo_epi32(a0_0, b1_0), _mm256_mullo_epi32(a0_1, b1_1)));

        __m256i a1_0 = _mm256_set1_epi32((int32_t)A[1 * lda + k + 0]);
        __m256i a1_1 = _mm256_set1_epi32((int32_t)A[1 * lda + k + 1]);
        c10 = _mm256_add_epi32(c10, _mm256_add_epi32(_mm256_mullo_epi32(a1_0, b0_0), _mm256_mullo_epi32(a1_1, b0_1)));
        c11 = _mm256_add_epi32(c11, _mm256_add_epi32(_mm256_mullo_epi32(a1_0, b1_0), _mm256_mullo_epi32(a1_1, b1_1)));

        __m256i a2_0 = _mm256_set1_epi32((int32_t)A[2 * lda + k + 0]);
        __m256i a2_1 = _mm256_set1_epi32((int32_t)A[2 * lda + k + 1]);
        c20 = _mm256_add_epi32(c20, _mm256_add_epi32(_mm256_mullo_epi32(a2_0, b0_0), _mm256_mullo_epi32(a2_1, b0_1)));
        c21 = _mm256_add_epi32(c21, _mm256_add_epi32(_mm256_mullo_epi32(a2_0, b1_0), _mm256_mullo_epi32(a2_1, b1_1)));

        __m256i a3_0 = _mm256_set1_epi32((int32_t)A[3 * lda + k + 0]);
        __m256i a3_1 = _mm256_set1_epi32((int32_t)A[3 * lda + k + 1]);
        c30 = _mm256_add_epi32(c30, _mm256_add_epi32(_mm256_mullo_epi32(a3_0, b0_0), _mm256_mullo_epi32(a3_1, b0_1)));
        c31 = _mm256_add_epi32(c31, _mm256_add_epi32(_mm256_mullo_epi32(a3_0, b1_0), _mm256_mullo_epi32(a3_1, b1_1)));
    }

    for (; k < K; ++k) {
        __m128i b_raw0 = _mm_loadl_epi64((const __m128i*)&B[k * ldb]);
        __m128i b_raw1 = _mm_loadl_epi64((const __m128i*)&B[k * ldb + 8]);

        __m256i b0 = _mm256_cvtepi8_epi32(b_raw0);
        __m256i b1 = _mm256_cvtepi8_epi32(b_raw1);

        __m256i a0 = _mm256_set1_epi32((int32_t)A[0 * lda + k]);
        c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0, b0));
        c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0, b1));

        __m256i a1 = _mm256_set1_epi32((int32_t)A[1 * lda + k]);
        c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(a1, b0));
        c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(a1, b1));

        __m256i a2 = _mm256_set1_epi32((int32_t)A[2 * lda + k]);
        c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(a2, b0));
        c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(a2, b1));

        __m256i a3 = _mm256_set1_epi32((int32_t)A[3 * lda + k]);
        c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(a3, b0));
        c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(a3, b1));
    }

    _mm256_storeu_si256((__m256i*)&C[0 * ldc + 0], c00);
    _mm256_storeu_si256((__m256i*)&C[0 * ldc + 8], c01);
    _mm256_storeu_si256((__m256i*)&C[1 * ldc + 0], c10);
    _mm256_storeu_si256((__m256i*)&C[1 * ldc + 8], c11);
    _mm256_storeu_si256((__m256i*)&C[2 * ldc + 0], c20);
    _mm256_storeu_si256((__m256i*)&C[2 * ldc + 8], c21);
    _mm256_storeu_si256((__m256i*)&C[3 * ldc + 0], c30);
    _mm256_storeu_si256((__m256i*)&C[3 * ldc + 8], c31);
}

NANOGEMM_AVX2_TARGET
static inline void sgemm_microkernel_4x8_i8_avx2(
    int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc)
{
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    for (int k = 0; k < K; ++k) {
        __m128i b_raw = _mm_loadl_epi64((const __m128i*)&B[k * ldb]);
        __m256i b = _mm256_cvtepi8_epi32(b_raw);

        __m256i a0 = _mm256_set1_epi32((int32_t)A[0 * lda + k]);
        c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a0, b));

        __m256i a1 = _mm256_set1_epi32((int32_t)A[1 * lda + k]);
        c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a1, b));

        __m256i a2 = _mm256_set1_epi32((int32_t)A[2 * lda + k]);
        c2 = _mm256_add_epi32(c2, _mm256_mullo_epi32(a2, b));

        __m256i a3 = _mm256_set1_epi32((int32_t)A[3 * lda + k]);
        c3 = _mm256_add_epi32(c3, _mm256_mullo_epi32(a3, b));
    }

    _mm256_storeu_si256((__m256i*)&C[0 * ldc], c0);
    _mm256_storeu_si256((__m256i*)&C[1 * ldc], c1);
    _mm256_storeu_si256((__m256i*)&C[2 * ldc], c2);
    _mm256_storeu_si256((__m256i*)&C[3 * ldc], c3);
}
#endif

#if defined(NANOGEMM_ARM_NEON)
static inline void sgemm_microkernel_4x8_i8_neon(
    int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc)
{
    int32x4_t c00 = vdupq_n_s32(0);
    int32x4_t c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0);
    int32x4_t c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0);
    int32x4_t c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0);
    int32x4_t c31 = vdupq_n_s32(0);

    for (int k = 0; k < K; ++k) {
        int8x8_t b8 = vld1_s8(&B[k * ldb]);
        int16x8_t b16 = vmovl_s8(b8);
        int32x4_t b_lo = vmovl_s16(vget_low_s16(b16));
        int32x4_t b_hi = vmovl_s16(vget_high_s16(b16));

        int32_t a0 = A[0 * lda + k];
        c00 = vmlaq_n_s32(c00, b_lo, a0);
        c01 = vmlaq_n_s32(c01, b_hi, a0);

        int32_t a1 = A[1 * lda + k];
        c10 = vmlaq_n_s32(c10, b_lo, a1);
        c11 = vmlaq_n_s32(c11, b_hi, a1);

        int32_t a2 = A[2 * lda + k];
        c20 = vmlaq_n_s32(c20, b_lo, a2);
        c21 = vmlaq_n_s32(c21, b_hi, a2);

        int32_t a3 = A[3 * lda + k];
        c30 = vmlaq_n_s32(c30, b_lo, a3);
        c31 = vmlaq_n_s32(c31, b_hi, a3);
    }

    vst1q_s32(&C[0 * ldc + 0], c00);
    vst1q_s32(&C[0 * ldc + 4], c01);
    vst1q_s32(&C[1 * ldc + 0], c10);
    vst1q_s32(&C[1 * ldc + 4], c11);
    vst1q_s32(&C[2 * ldc + 0], c20);
    vst1q_s32(&C[2 * ldc + 4], c21);
    vst1q_s32(&C[3 * ldc + 0], c30);
    vst1q_s32(&C[3 * ldc + 4], c31);
}
#endif

NANOGEMM_API void nanogemm_gemm_i8i8i32(
    int M, int N, int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

#if defined(NANOGEMM_X86_AVX2)
    int i = 0;
    for (; i <= M - 4; i += 4) {
        int j = 0;
        for (; j <= N - 16; j += 16) {
            sgemm_microkernel_4x16_i8_avx2(K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
        }
        for (; j <= N - 8; j += 8) {
            sgemm_microkernel_4x8_i8_avx2(K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
        }
        if (j < N) {
            sgemm_i8_edge(4, N - j, K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
        }
    }
    if (i < M) {
        sgemm_i8_edge(M - i, N, K, &A[i * lda], lda, &B[0], ldb, &C[i * ldc], ldc);
    }
#elif defined(NANOGEMM_ARM_NEON)
    int i = 0;
    for (; i <= M - 4; i += 4) {
        int j = 0;
        for (; j <= N - 8; j += 8) {
            sgemm_microkernel_4x8_i8_neon(K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
        }
        if (j < N) {
            sgemm_i8_edge(4, N - j, K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
        }
    }
    if (i < M) {
        sgemm_i8_edge(M - i, N, K, &A[i * lda], lda, &B[0], ldb, &C[i * ldc], ldc);
    }
#else
    sgemm_i8_edge(M, N, K, A, lda, B, ldb, C, ldc);
#endif
}
