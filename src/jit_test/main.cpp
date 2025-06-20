#include <arm_mve.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include "LPRTC.hpp"
#include "board.h"
#include "dsp/matrix_functions.h"
#include "generators/Gemm.hpp"
#include "fault_handler.h"
#include "profiling.hpp"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"
#include "benchmark.hpp"
#include "timing.hpp"
#include "arm_math.h"

#ifdef M55_HE
constexpr float peak = 0.64;
#endif

#ifdef M55_HP
constexpr float peak = 1.6;
#endif

constexpr bool testReference = false;
constexpr bool testIntrinsics = false;
constexpr bool testArm = false;
constexpr bool testJitter = true;

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

    // Vektorregister initialisieren mit vdup
    // c00_vreg = vdupq_n_f32(0.0f);
    // c01_vreg = vdupq_n_f32(0.0f);
    // c10_vreg = vdupq_n_f32(0.0f);
    // c11_vreg = vdupq_n_f32(0.0f);
    // c20_vreg = vdupq_n_f32(0.0f);
    // c21_vreg = vdupq_n_f32(0.0f);

    c00_vreg = vld1q_f32(&c[0]);
    c01_vreg = vld1q_f32(&c[4]);
    c10_vreg = vld1q_f32(&c[ldc]);
    c11_vreg = vld1q_f32(&c[ldc + 4]);
    c20_vreg = vld1q_f32(&c[2 * ldc]);
    c21_vreg = vld1q_f32(&c[2 * ldc + 4]);

    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*lda]);
        a1p_vreg = vld1q_f32(&a[p*lda+4]);

        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);

        c00_vreg = vfmaq_f32(c00_vreg, a0p_vreg, bp0_vreg);
        c01_vreg = vfmaq_f32(c01_vreg, a1p_vreg, bp0_vreg);
        c10_vreg = vfmaq_f32(c10_vreg, a0p_vreg, bp1_vreg);
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

extern "C" {
    void gemm_20x24_jit(float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ c);
    void gemm_20x24_tuned(float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ c);
}

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));
/*constexpr uint32_t peakCount = 50000;
static float a[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float b[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float c[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
*/
// #define CONST_SIZE
#ifdef CONST_SIZE
static const uint32_t M = 24;
static const uint32_t K = 24;
static const uint32_t N = 24;
static float bigA[M*K];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[K*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[M*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[M*N];// __attribute__((used, section(".bss.array_region_sram0")));
#else
static constexpr uint32_t arrayMaxSize = 240;
static float bigA[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
#endif

JIT::Instructions::Instruction16 globalBuffer[3072] __attribute__((section(".itcm_jit"), aligned(4)));
// JIT::Instructions::Instruction16 globalBuffer[3072] __attribute__((aligned(4)));

void initMatrices(float * a, float * b, float * c, float * cref, const uint32_t m, const uint32_t n, const uint32_t k, bool zeroC = false) {
    for (uint32_t i = 0; i < m*k; i++) a[i] = i;
    for (uint32_t i = 0; i < k*n; i++) b[i] = i + 1;
    for (uint32_t i = 0; i < m*n; i++) c[i] = zeroC ? 0 : i + 2;
	for (uint32_t i = 0; i < m*n; i++) cref[i] = zeroC ? 0 : i + 2;
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

int32_t testShapeGenerateTime(uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator) {
    JIT::Generators::Gemm::Func gemmFunc;
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);

    auto start = RTC_Clock::now();
    for (uint32_t it = 0; it < iterations; it++) {
        gemmFunc = generator.generate(m, k, n, m, k, m);
        gemmFunc = generator.bufferToFunc(globalBuffer);
    }
    auto end = RTC_Clock::now();
    gemmFunc(bigA, bigB, bigC);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return time; // return negative value if test not succesful

}

int32_t testShape(uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator) {
    auto gemmFunc = generator.generate(m, k, n, m, k, m);
    // auto gemmFunc = generator.generate(mBlocking, kBlocking, n, m, k, m);
    gemmFunc = generator.bufferToFunc(globalBuffer);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // setupProfilingMemory();
    // startCounting();
    gemmFunc(bigA, bigB, bigC);
    // jitBlocked_mk(bigA, bigB, bigC, m, n, k, gemmFunc);
    // stopCounting();
    // printCounterMemory();

    gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    int32_t compareResult = compare(bigC, bigCRef, m*n);
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    auto start = RTC_Clock::now();
    for (uint32_t it = 0; it < iterations; it++) {
        // jitBlocked_mk(bigA, bigB, bigC, m, n, k, gemmFunc);
        gemmFunc(bigA, bigB, bigC);
    }
    auto end = RTC_Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (compareResult != -1) SEGGER_RTT_printf(0, "Fail at %d;", compareResult);
    return compareResult != -1 ? -time : time; // return negative value if test not succesful
}

int32_t testShapeReference(uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    }
    auto end = RTC_Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int32_t testShapeIntrinsics(uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    gemm_intrinsics_8x3(bigA, bigB, bigC, n, k, m, m, k, m);
    gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    int32_t compareResult = compare(bigC, bigCRef, m*n);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_intrinsics_8x3(bigA, bigB, bigCRef, n, k, m, m, k, m);
    }
    auto end = RTC_Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return compareResult != -1 ? -time : time; // return negative value if test not succesful
}


int32_t testShapeArm(uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    arm_matrix_instance_f32 armA;
    arm_matrix_instance_f32 armB;
    arm_matrix_instance_f32 armC;
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k, true);
    arm_mat_init_f32(&armA, m, k, bigA);
    arm_mat_init_f32(&armB, k, n, bigB);
    arm_mat_init_f32(&armC, m, n, bigC);
    gemm_reference_row_major(bigA, bigB, bigCRef, n, k, m, k, n, n);
    arm_mat_mult_f32(&armA, &armB, &armC);
    int32_t compareResult = compare(bigC, bigCRef, m*n);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k, true);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        arm_mat_mult_f32(&armA, &armB, &armC);
    }
    auto end = RTC_Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return compareResult != -1 ? -time : time; // return negative value if test not succesful
}

void testSquareShapes() {
    int32_t time;
    double gflops;
    uint32_t m, n, k;
    JIT::Generators::Gemm gemmGen;
    SEGGER_RTT_printf(0, "--- START TEST SQUARE SHAPES ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 2; i < 240; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        m = i;
        n = i;
        k = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        iterations = 30000;
        if (testArm) {
            time = testShapeArm(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShapeGenerateTime(m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;TillJIT-04.06.;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
        
        if (testIntrinsics && m % 8 == 0 && n % 3 == 0) {
            time = testShapeIntrinsics(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;Intrinsics;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST SQUARE SHAPES ---\n\n");
}

void testGrowingK() {
    int32_t time;
    double gflops;
    uint32_t m = 24, n = 24, k;
    JIT::Generators::Gemm gemmGen;
    SEGGER_RTT_printf(0, "--- START TEST GROWING K ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 2; i < 240; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        k = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        if (testArm) {
            time = testShapeArm(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;TillJIT-04.06.;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics) {
            time = testShapeIntrinsics(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING K ---\n\n");
}

void testGrowingM() {
    int32_t time;
    double gflops;
    uint32_t m = 1, n = 24, k = 24;
    JIT::Generators::Gemm gemmGen;
    SEGGER_RTT_printf(0, "--- START TEST GROWING M ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 2; i < 240; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        m = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        if (testArm) {
            time = testShapeArm(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;TillJIT-04.06.;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics && m % 8 == 0) {
            time = testShapeIntrinsics(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING M ---\n\n");
}

void testGrowingN() {
    int32_t time;
    double gflops;
    uint32_t m = 24, n = 1, k = 24;
    JIT::Generators::Gemm gemmGen;
    SEGGER_RTT_printf(0, "--- START TEST GROWING N ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 2; i < 240; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        n = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        if (testArm) {
            time = testShapeArm(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;TillJIT-04.06.;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics && n % 3 == 0) {
            time = testShapeIntrinsics(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING N ---\n\n");
}

void constSizeTest(uint32_t m, uint32_t n, uint32_t k) {
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    JIT::Generators::Gemm gemmGen;
    uint32_t repeats = 5;
    uint32_t flops = 2 * m * k * n;
    uint32_t iterations = (peak * pow(10, 9)) / flops;
    int32_t time;
    double gflops;
    for (uint32_t i = 0; i < repeats; i++) {
        time = testShape(m, n, k, iterations, gemmGen);
        gflops = (flops / (time/1000.0f * pow(10, 9))) * iterations;
        sprintf(PRINTF_OUT_STRING, "Unroll5;%d;%d;%d;TillJIT;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    }
}

void configureMPU() {
    // Disable MPU
    ARM_MPU_Disable();
    
    // Define the MPU region table
    static const ARM_MPU_Region_t mpu_table[] = {
        {
            // SRAM0 region with caching enabled
            .RBAR = ARM_MPU_RBAR(0x02000000UL, ARM_MPU_SH_NON, 0UL, 1UL, 0UL), // RW, NP, XN
            .RLAR = ARM_MPU_RLAR(0x023FFFFFUL, 1UL) // SRAM0 with cacheable attribute
        }
    };
    
    // Define memory attributes
    ARM_MPU_SetMemAttr(0UL, ARM_MPU_ATTR_DEVICE); // Device Memory
    ARM_MPU_SetMemAttr(1UL, ARM_MPU_ATTR( // Normal Memory, Write-back, Read/Write-Allocate
        ARM_MPU_ATTR_MEMORY_(1,1,1,1), 
        ARM_MPU_ATTR_MEMORY_(1,1,1,1)
    ));
    ARM_MPU_SetMemAttr(2UL, ARM_MPU_ATTR( // Normal Memory, Transient, Write-through, Read-Allocate
        ARM_MPU_ATTR_MEMORY_(0,0,1,0), 
        ARM_MPU_ATTR_MEMORY_(0,0,1,0)
    ));
    
    // Load the regions from the table
    ARM_MPU_Load(0U, &mpu_table[0], sizeof(mpu_table)/sizeof(ARM_MPU_Region_t));
    
    // Enable MPU with default memory map for privileged access
    ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);
}

__NO_RETURN int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

    // configureMPU();

    // SCB_InvalidateICache();
    // SCB_EnableICache();

    // SCB_InvalidateDCache();
    // SCB_EnableDCache();

#ifdef CONST_SIZE
    uint32_t m = M, n = N, k = K;
	initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    uint32_t flops = 2 * m * k * n;
    uint32_t iterations = (1.6 * pow(10, 9)) / flops;
    int32_t time;
    double gflops;

    constSizeTest(m, n, k);

    // initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // gemm_20x24_jit(bigA, bigB, bigC);
    // gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    // int32_t compareResult = compare(bigC, bigCRef, m*n);
    // initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // auto start = RTC_Clock::now();
    // for (uint32_t j = 0; j < iterations; j++) {
    //     gemm_20x24_jit(bigA, bigB, bigC);
    // }
    // auto end = RTC_Clock::now();
    // time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // gflops = (flops / (time/1000.0f * pow(10, 9))) * iterations;
    // sprintf(PRINTF_OUT_STRING, "ASM 20x24;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, compareResult);
    // SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    // initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    // gemm_20x24_tuned(bigA, bigB, bigC);
    // compareResult = compare(bigC, bigCRef, m*n);
    // // SEGGER_RTT_printf(0, "Result: %d", compareResult);
    // initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // start = RTC_Clock::now();
    // for (uint32_t j = 0; j < iterations; j++) {
    //     gemm_20x24_tuned(bigA, bigB, bigC);
    // }
    // end = RTC_Clock::now();
    // time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // gflops = (flops / (time/1000.0f * pow(10, 9))) * iterations;
    // sprintf(PRINTF_OUT_STRING, "ASM 20x24 Tuned;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, compareResult);
    // SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


    // time = testShapeArm(m, n, k, iterations);
    // gflops = (flops / (time/1000.0f * pow(10, 9))) * iterations;
    // sprintf(PRINTF_OUT_STRING, "ARM-CMSIS-DSP %dx%dx%d (%d): %f, %f, %s\r\n", m, k, n, time, bigC[0], gflops, time < 0 ? "Error" : "Correct");
    // SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
/*
    time = testShape(m, n, k, iterations, gemmGen);
    gflops = (flops / (time/1000.0f * pow(10, 9))) * iterations;
    sprintf(PRINTF_OUT_STRING, "TillJIT %dx%dx%d (%d): %f, %f, %s\r\n", m, k, n, time, bigC[0], gflops, time < 0 ? "Error" : "Correct");
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
*/
#endif
#ifndef CONST_SIZE
    testSquareShapes();
    // testGrowingK();
    // testGrowingM();
    // testGrowingN();
#endif
	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
