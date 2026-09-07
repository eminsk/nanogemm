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
/* -------------------------------------------------------------
 * AVX2 + FMA 6x16 Microkernel
 * Computes a 6x16 tile of C from 6xK of A and Kx16 of B
 * ------------------------------------------------------------- */
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
#endif


/* -------------------------------------------------------------
 * Fallback Edge Kernel (handles remainders where M<6 or N<16)
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

            #if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 va = _mm256_loadu_ps(&A[i * lda + k]);
                __m256 vb = _mm256_set_ps(
                    B[(k + 7) * ldb + j], B[(k + 6) * ldb + j],
                    B[(k + 5) * ldb + j], B[(k + 4) * ldb + j],
                    B[(k + 3) * ldb + j], B[(k + 2) * ldb + j],
                    B[(k + 1) * ldb + j], B[(k + 0) * ldb + j]);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            #endif

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
                    if (j < n_block) {
                        sgemm_edge_kernel(
                            6, n_block - j, k_block,
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
