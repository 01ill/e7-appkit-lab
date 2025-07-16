#include "gemm_kernel.hpp"
#include <cstdint>
#include "arm_math.h"
#include "../backend/Backend.hpp"
#include "../generators/Gemm.hpp"

void gemm_reference_row_major(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j++) { // j = n
        for (uint32_t i = 0; i < m; i++) { // i = m
            for (uint32_t p = 0; p < k; p++) { // p = k
                c[i * ldc + j] += a[i * lda + p] * b[p * ldb + j];
            }
        }
    }
}


void gemm_reference_column_major(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j++) { // j = n
        for (uint32_t i = 0; i < m; i++) { // i = m
            for (uint32_t p = 0; p < k; p++) { // p = k
                c[j * ldc + i] += a[p * lda + i] * b[j * ldb + p];
            }
        }
    }
}

void addDot8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c00_vreg, c01_vreg, c10_vreg, c11_vreg, c20_vreg, c21_vreg, a0p_vreg, a1p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg;
    const float32_t *bp0, *bp1, *bp2;

    bp0 = &b[0];
    bp1 = &b[ldb];
    bp2 = &b[2*ldb];

    c00_vreg = vld1q_f32(&c[0]);
    c01_vreg = vld1q_f32(&c[4]);
    c10_vreg = vld1q_f32(&c[ldc]);
    c11_vreg = vld1q_f32(&c[ldc + 4]);
    c20_vreg = vld1q_f32(&c[2 * ldc]);
    c21_vreg = vld1q_f32(&c[2 * ldc + 4]);

    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*lda]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        c00_vreg = vfmaq_f32(c00_vreg, a0p_vreg, bp0_vreg);
        a1p_vreg = vld1q_f32(&a[p*lda+4]);
        c01_vreg = vfmaq_f32(c01_vreg, a1p_vreg, bp0_vreg);
        bp1_vreg = vdupq_n_f32(*bp1++);
        c10_vreg = vfmaq_f32(c10_vreg, a0p_vreg, bp1_vreg);
        bp2_vreg = vdupq_n_f32(*bp2++);
        c11_vreg = vfmaq_f32(c11_vreg, a1p_vreg, bp1_vreg);
        c20_vreg = vfmaq_f32(c20_vreg, a0p_vreg, bp2_vreg);
        c21_vreg = vfmaq_f32(c21_vreg, a1p_vreg, bp2_vreg);
    }

    vst1q_f32(&c[0], c00_vreg);
    vst1q_f32(&c[4], c01_vreg);

    vst1q_f32(&c[ldc], c10_vreg);
    vst1q_f32(&c[ldc+4], c11_vreg);

    vst1q_f32(&c[2*ldc], c20_vreg);
    vst1q_f32(&c[2*ldc+4], c21_vreg);
}

void gemm_intrinsics_8x3(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j += 3) { // j = n
        for (uint32_t i = 0; i < m; i += 8) { // i = m
            addDot8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(&a[i], &b[j * ldb], &c[j * ldc + i], n, k, m, lda, ldb, ldc);
        }
    }
}

void addDot4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, c4_vreg, c5_vreg, c6_vreg, a0p_vreg, bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg, bp4_vreg, bp5_vreg, bp6_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3, *bp4, *bp5, *bp6;

    bp0 = &b[0];
    bp1 = &b[ldb];
    bp2 = &b[2*ldb];
    bp3 = &b[3*ldb];
    bp4 = &b[4*ldb];
    bp5 = &b[5*ldb];
    // bp6 = &b[6*ldb];

    c0_vreg = vld1q_f32(&c[0]);
    c1_vreg = vld1q_f32(&c[ldc]);
    c2_vreg = vld1q_f32(&c[2 * ldc]);
    c3_vreg = vld1q_f32(&c[3 * ldc]);
    c4_vreg = vld1q_f32(&c[4 * ldc]);
    c5_vreg = vld1q_f32(&c[5 * ldc]);
    // c6_vreg = vld1q_f32(&c[6 * ldc]);

    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*lda]);

        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);
        bp4_vreg = vdupq_n_f32(*bp4++);
        bp5_vreg = vdupq_n_f32(*bp5++);
        // bp6_vreg = vdupq_n_f32(*bp6++);


        c0_vreg = vfmaq_f32(c0_vreg, a0p_vreg, bp0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, a0p_vreg, bp1_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, a0p_vreg, bp2_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, a0p_vreg, bp3_vreg);
        c4_vreg = vfmaq_f32(c4_vreg, a0p_vreg, bp4_vreg);
        c5_vreg = vfmaq_f32(c5_vreg, a0p_vreg, bp5_vreg);
        // c6_vreg = vfmaq_f32(c6_vreg, a0p_vreg, bp6_vreg);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[ldc], c1_vreg);
    vst1q_f32(&c[2*ldc], c2_vreg);
    vst1q_f32(&c[3*ldc], c3_vreg);
    vst1q_f32(&c[4*ldc], c4_vreg);
    vst1q_f32(&c[5*ldc], c5_vreg);
    // vst1q_f32(&c[6*ldc], c6_vreg);
}

void gemm_intrinsics_4x6(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j += 6) { // j = n
        for (uint32_t i = 0; i < m; i += 4) { // i = m
            addDot4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(&a[i], &b[j * ldb], &c[j * ldc + i], n, k, m, lda, ldb, ldc);
        }
    }
}

void addDot16x1_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, a0p_vreg, a1p_vreg, a2p_vreg, a3p_vreg, bp0_vreg;
    const float32_t *bp0;

    bp0 = &b[0];

    c0_vreg = vld1q_f32(&c[0]);
    c1_vreg = vld1q_f32(&c[4]);
    c2_vreg = vld1q_f32(&c[8]);
    c3_vreg = vld1q_f32(&c[12]);

    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*lda]);
        a1p_vreg = vld1q_f32(&a[p*lda + 4]);
        a2p_vreg = vld1q_f32(&a[p*lda + 8]);
        a3p_vreg = vld1q_f32(&a[p*lda + 12]);

        bp0_vreg = vdupq_n_f32(*bp0++);


        c0_vreg = vfmaq_f32(c0_vreg, a0p_vreg, bp0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, a1p_vreg, bp0_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, a2p_vreg, bp0_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, a3p_vreg, bp0_vreg);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[4], c1_vreg);
    vst1q_f32(&c[8], c2_vreg);
    vst1q_f32(&c[12], c3_vreg);
    // vst1q_f32(&c[6*ldc], c6_vreg);
}

void gemm_intrinsics_16x1(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j += 1) { // j = n
        for (uint32_t i = 0; i < m; i += 16) { // i = m
            addDot16x1_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(&a[i], &b[j * ldb], &c[j * ldc + i], n, k, m, lda, ldb, ldc);
        }
    }
}

constexpr uint32_t mBlocking = 24;
constexpr uint32_t nBlocking = 24;
constexpr uint32_t kBlocking = 24;

void jitBlocked(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc) {   
    for (uint32_t i = 0; i < m; i += mBlocking) {
        for (uint32_t j = 0; j < n; j += nBlocking) {
            for (uint32_t p = 0; p < k; p += kBlocking) {
                gemmFunc(&a[i * k + p], &b[j * k + p], &c[j * m + i]);
            }
        }
    }
}

void jitBlocked_m(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc) {   
    for (uint32_t i = 0; i < m; i += mBlocking) {
        gemmFunc(&a[i], &b[0], &c[i]);
    }
}

void jitBlocked_mk(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc) {   
    for (uint32_t i = 0; i < m; i += mBlocking) {
        for (uint32_t p = 0; p < k; p += kBlocking) {
            gemmFunc(&a[p * m + i], &b[p], &c[i]);
        }
    }
}

void jitBlocked_mkn(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc) {   
    for (uint32_t i = 0; i < m; i += mBlocking) {
        for (uint32_t p = 0; p < k; p += kBlocking) {
            for (uint32_t j = 0; j < n; j += nBlocking) {
                gemmFunc(&a[p * m + i], &b[j * k + p], &c[j * m + i]);
            }
        }
    }
}

/*
void armBlocked(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc) {   
    for (uint32_t i = 0; i < m; i += mBlocking) {
        for (uint32_t j = 0; j < n; j += nBlocking) {                
            arm_mat_mult_f32(&a[i], &b[j * k ], &c[j * m + i]);
        }
    }
}
*/
