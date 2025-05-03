/* Copyright (C) 2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
*/
// #include <arm_mve_types.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ratio>

#include "board.h"
#include "RTE_Components.h"
#include "m-profile/cmsis_armclang_m.h"
#include "profiling.hpp"
#include CMSIS_device_header

#include "fault_handler.h"

#include "benchmark.hpp"
#include "timing.hpp"
#include <arm_mve.h>
#include "arm_math.h"
#include "SEGGER_RTT.h"

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));


static constexpr uint32_t arrayMaxSize = 24;
static float32_t bigA[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigB[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigC[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigCRef[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t packedA[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t packedB[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));

extern "C" {
    float32_t gemm_4x6(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len_k);
    float32_t gemm_4x4(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len_k);
    float32_t gemm_asm_4x4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_4x6(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_8x3(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_8x3_microkernel(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_24x24(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_4x7(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len);
}

/*
M=N=K=len A: mxk, B: kxn, C: mxn
Cij = sum Aik * Bkj
*/
float32_t gemm_reference(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) { // i = m
        for (uint32_t j = 0; j < len; j++) { // j = n
            for (uint32_t k = 0; k < len; k++) { // k = k
                c[i * len + j] += a[i * len + k] * b[k * len + j];
            }
        }
    }
    return c[0];
}


/*
M=N=K=len A: mxk, B: kxn, C: mxn
Cij = sum Aik * Bkj
*/
float32_t gemm_reference_column_major(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j++) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            // c[j * len + i] = 0.0f;
            for (uint32_t k = 0; k < len; k++) { // k = k
                c[j * len + i] += a[k * len + i] * b[j * len + k];
            }
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_10
*/
void addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, a0p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];

    // Vektorregister initialisieren mit vdup
    c0_vreg = vdupq_n_f32(0.0f);
    c1_vreg = vdupq_n_f32(0.0f);
    c2_vreg = vdupq_n_f32(0.0f);
    c3_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);

        c0_vreg = vfmaq_f32(c0_vreg, a0p_vreg, bp0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, a0p_vreg, bp1_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, a0p_vreg, bp2_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, a0p_vreg, bp3_vreg);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[len], c1_vreg);
    vst1q_f32(&c[len*2], c2_vreg);
    vst1q_f32(&c[len*3], c3_vreg);
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_11
*/
void inner_kernel_4x4_intrinsics(uint32_t m, uint32_t n, uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, uint32_t len) {
    for (uint32_t j = 0; j < n; j += 4) {
        for (uint32_t i = 0; i < m; i += 4) {
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
}

constexpr uint32_t mc = 16;
constexpr uint32_t kc = 16;

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics(ib, len, pb, &a[p*len + i], &b[p], &c[i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_10
*/
void addDot4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, c4_vreg, c5_vreg, a0p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg, bp4_vreg, bp5_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3, *bp4, *bp5;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];
    bp4 = &b[4*len];
    bp5 = &b[5*len];

    // Vektorregister initialisieren mit vdup
    c0_vreg = vdupq_n_f32(0.0f);
    c1_vreg = vdupq_n_f32(0.0f);
    c2_vreg = vdupq_n_f32(0.0f);
    c3_vreg = vdupq_n_f32(0.0f);
    c4_vreg = vdupq_n_f32(0.0f);
    c5_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);
        bp4_vreg = vdupq_n_f32(*bp4++);
        bp5_vreg = vdupq_n_f32(*bp5++);

        c0_vreg = vfmaq_f32(c0_vreg, a0p_vreg, bp0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, a0p_vreg, bp1_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, a0p_vreg, bp2_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, a0p_vreg, bp3_vreg);
        c4_vreg = vfmaq_f32(c4_vreg, a0p_vreg, bp4_vreg);
        c5_vreg = vfmaq_f32(c5_vreg, a0p_vreg, bp5_vreg);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[len], c1_vreg);
    vst1q_f32(&c[len*2], c2_vreg);
    vst1q_f32(&c[len*3], c3_vreg);
    vst1q_f32(&c[len*4], c4_vreg);
    vst1q_f32(&c[len*5], c5_vreg);
}

float32_t gemm_cm_dot_unroll4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 6) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

void addDot8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c00_vreg, c01_vreg, c10_vreg, c11_vreg, c20_vreg, c21_vreg, a0p_vreg, a1p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg;
    const float32_t *bp0, *bp1, *bp2;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];

    // Vektorregister initialisieren mit vdup
    c00_vreg = vdupq_n_f32(0.0f);
    c01_vreg = vdupq_n_f32(0.0f);
    c10_vreg = vdupq_n_f32(0.0f);
    c11_vreg = vdupq_n_f32(0.0f);
    c20_vreg = vdupq_n_f32(0.0f);
    c21_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        a1p_vreg = vld1q_f32(&a[p*len+4]);

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

    vst1q_f32(&c[len], c10_vreg);
    vst1q_f32(&c[len+4], c11_vreg);

    vst1q_f32(&c[2*len], c20_vreg);
    vst1q_f32(&c[2*len+4], c21_vreg);
}

float32_t gemm_cm_dot_unroll8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 3) { // j = n
        for (uint32_t i = 0; i < len; i += 8) { // i = m
            addDot8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

float32_t frame_gemm_8x3(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j+= 3) {
        for (uint32_t i = 0; i < len; i += 8) {
            gemm_asm_8x3_microkernel(&a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

__NO_RETURN int main (void) {
    fault_dump_enable(true);
    SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
    setupTests();

    uint32_t iterations = 8000;
    uint32_t time;
    float32_t gflops;
    float32_t result = 0.0f;

    arm_matrix_instance_f32 armA;
    arm_matrix_instance_f32 armB;
    arm_matrix_instance_f32 armC;
    arm_status status;

    // for (uint32_t i = 24; i <= arrayMaxSize; i += 24) {
    //     uint32_t flops = iterations * 2 * pow(i,3);
    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     arm_mat_init_f32(&armA, i, i, bigA);
    //     arm_mat_init_f32(&armB, i, i, bigB);
    //     arm_mat_init_f32(&armC, i, i, bigC);

    //     time = benchmarkArm(arm_mat_mult_f32, iterations, &status, &armA, &armB, &armC);
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "CMSIS-DSP %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     time = benchmark(gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics, iterations, &result, bigA, bigB, bigC, i);
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM Intrinsics 4x4 %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     time = benchmark(gemm_cm_dot_unroll4x6_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics, iterations, &result, bigA, bigB, bigC, i);
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM Intrinsics 4x6 %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    //     RTC_Clock::time_point start = RTC_Clock::now();
    //     RTC_Clock::time_point end = RTC_Clock::now();

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     /*RTC_Clock::time_point start = RTC_Clock::now();
    //     for (uint32_t j = 0; j < iterations; j++) {
    //         gemm_asm_4x4(bigA, bigB, bigC, i);
    //     }
    //     RTC_Clock::time_point end = RTC_Clock::now();*/
    //     time = benchmark(gemm_asm_4x4, iterations, &result, bigA, bigB, bigC, i);
    //     //time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM 4x4 ASM %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     start = RTC_Clock::now();
    //     for (uint32_t j = 0; j < iterations; j++) {
    //         gemm_asm_4x6(bigA, bigB, bigC, i);
    //     }
    //     end = RTC_Clock::now();
    //     /*setupProfilingMVEStalls();
    //     startCounting();
    //     gemm_asm_4x6(bigA, bigB, bigC, i);
    //     stopCounting();*/
    //     // printCounter();

    //     time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM 4x6 ASM %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     start = RTC_Clock::now();
    //     for (uint32_t j = 0; j < iterations; j++) {
    //         gemm_asm_8x3(bigA, bigB, bigC, i);
    //     }
    //     end = RTC_Clock::now();

    //     time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM 8x3 ASM %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     start = RTC_Clock::now();
    //     for (uint32_t j = 0; j < iterations; j++) {
    //         frame_gemm_8x3(bigA, bigB, bigC, i);
    //     }
    //     end = RTC_Clock::now();

    //     time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM 8x3 Microkernel %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    //     for (uint32_t j = 0; j < i*i; j++) {
    //         bigA[j] = j;
    //         bigB[j] = j;
    //         bigC[j] = 0;
    //     }
    //     start = RTC_Clock::now();
    //     for (uint32_t j = 0; j < iterations; j++) {
    //         gemm_cm_dot_unroll8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(bigA, bigB, bigC, i);
    //     }
    //     end = RTC_Clock::now();

    //     time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    //     sprintf(PRINTF_OUT_STRING, "GEMM CM 8x3 Intrinsics %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
    //     SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    //     //iterations = iterations >> 2U; // Iterationen müssen weniger werden, sonst rechnet der nie fertig
    //     iterations >>= 1U;
    // }
    // SEGGER_RTT_printf(0, "Fertig!\n"); 
    uint32_t flops = iterations * 2 * pow(arrayMaxSize,3);


    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
    }
    RTC_Clock::time_point start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_asm_8x3(bigA, bigB, bigC, arrayMaxSize);
    }
    RTC_Clock::time_point end = RTC_Clock::now();

    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "GEMM CM 8x3 ASM %dx%d (%d): %f, %f, %f\r\n", arrayMaxSize, arrayMaxSize, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
    }
    start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_asm_24x24(bigA, bigB, bigC, arrayMaxSize);
    }
    end = RTC_Clock::now();
    setupProfilingMVEStalls();
    startCounting();
    gemm_asm_24x24(bigA, bigB, bigC, arrayMaxSize);
    stopCounting();
    printCounter();

    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "GEMM CM 24x24 ASM %dx%d (%d): %f, %f, %f\r\n", arrayMaxSize, arrayMaxSize, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
    }
    start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_cm_dot_unroll8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(bigA, bigB, bigC, arrayMaxSize);
    }
    end = RTC_Clock::now();

    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "GEMM CM 8x3 Intrinsics %dx%d (%d): %f, %f, %f\r\n", arrayMaxSize, arrayMaxSize, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
/*

    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }

    arm_mat_init_f32(&armA, arrayMaxSize, arrayMaxSize, bigA);
    arm_mat_init_f32(&armB, arrayMaxSize, arrayMaxSize, bigB);
    arm_mat_init_f32(&armC, arrayMaxSize, arrayMaxSize, bigC);

    arm_mat_mult_f32(&armA, &armB, &armC);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    int32_t compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "ARM CMSIS-DSP: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "ARM CMSIS-DSP: Test nicht erfolgreich bei %d\n", compareResult);
    }


    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm_asm_4x6(bigA, bigB, bigC, arrayMaxSize);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "GEMM-ASM 4x6: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "GEMM-ASM 4x6: Test nicht erfolgreich bei %d\n", compareResult);
    }

    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(bigA, bigB, bigC, arrayMaxSize);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "4x4 Intrinsics: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "4x4 Intrinsics: Test nicht erfolgreich bei %d\n", compareResult);
    }

    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm_asm_8x3(bigA, bigB, bigC, arrayMaxSize);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "GEMM-ASM 8x3: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "GEMM-ASM 8x3: Test nicht erfolgreich bei %d\n", compareResult);
    }

    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }

    gemm_cm_dot_unroll8x3_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(bigA, bigB, bigC, arrayMaxSize);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "8x3 Intrinsics: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "8x3 Intrinsics: Test nicht erfolgreich bei %d\n", compareResult);
        sprintf(PRINTF_OUT_STRING, "a: %f, b: %f, c: %f, cRef: %f\n", bigA[385], bigB[385], bigC[385], bigCRef[385]);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    
    }
*/
    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm_asm_24x24(bigA, bigB, bigC, arrayMaxSize);
    gemm_reference_column_major(bigA, bigB, bigCRef, arrayMaxSize);
    int32_t compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "GEMM-ASM 24x24: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "GEMM-ASM 24x24: Test nicht erfolgreich bei %d\n", compareResult);
    }

    stopTests();

    while (1) __WFE();
}
