#ifndef GEMM_KERNEL_HPP
#define GEMM_KERNEL_HPP
#include <cstdint>
#include "../generators/Gemm.hpp"

void gemm_reference_row_major(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
void gemm_reference_column_major(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
// void addDot8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
void gemm_intrinsics_8x3(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
// void addDot4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
void gemm_intrinsics_4x6(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
// void addDot16x1_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
void gemm_intrinsics_16x1(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc);
void jitBlocked(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc);
void jitBlocked_m(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc);
void jitBlocked_mk(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc);
void jitBlocked_mkn(float * a, float * b, float * c, const uint32_t m, const uint32_t n, const uint32_t k, JIT::Generators::Gemm::Func gemmFunc);

#endif // GEMM_KERNEL_HPP