#include <cstdint>
#include <stdio.h>

#include "board.h"
#include "RTE_Components.h"
#include CMSIS_device_header

#include "fault_handler.h"

#include "benchmark.hpp"
#include "timing.hpp"
#include <arm_mve.h>

extern "C" {
    float32_t dotp_scalar(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len);
    float32_t dotp_mve(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len);
}

float32_t dotp_reference(const float32_t *__restrict__ a, const float32_t *__restrict__ b, float32_t * c, const uint32_t len) {
    float32_t sum = 0.0;
    for (uint32_t i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    *c = sum;
    return sum;
}

float32_t dotp_mve_intrinsic(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len) {
    float32x4_t vecA, vecB, vecSum;
    uint32_t blockCount;
    float32_t sum = 0.0f;
    vecSum = vdupq_n_f32(0.0f); // init vecSum = {0, 0, 0, 0}

    blockCount = len >> 2U; // 4 floats pro Durchlauf -> Teilen durch 4
    while (blockCount > 0U) {
        vecA = vld1q(a); // 4 F32 laden
        a += 4; // 4 F32 weiter
        vecB = vld1q(b); // 4 F32 laden
        b += 4; // 4 F32 weiter

        vecSum = vfmaq(vecSum, vecA, vecB); // vecSum += vecA * vecB
        blockCount--;
    }

    // len % 4 != 0
    blockCount = len & 3U; // Rest
    if (blockCount > 0U) {
        mve_pred16_t p0 = vctp32q(blockCount); // Predicate
        vecA = vld1q(a);
        vecB = vld1q(b);
        vecSum = vfmaq_m(vecSum, vecA, vecB, p0); // Predicated MAC
    }

    // Summenvektor zusammenaddieren
    // sum = vecAddAcrossF32Mve(vecSum);
    sum = vgetq_lane_f32(vecSum, 0U) + vgetq_lane_f32(vecSum, 1U) + vgetq_lane_f32(vecSum, 2U) + vgetq_lane_f32(vecSum, 3U);
    
    *c = sum;
    return sum;
}

float32_t dotp_scalar_intrinsic(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len) {
    float32_t sum = 0;
    for (uint32_t i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    *c = sum;
    return sum;
}

int main (void)
{
    // Init pinmux using boardlib
    // BOARD_Pinmux_Init();

    fault_dump_enable(true);

    setupTests();

    // SysTick_Config(SystemCoreClock / 10);

    printf("\r\nHello World!\r\n");

    const uint32_t len = 128;
    const uint32_t iterations = 100000;
    float32_t a[len];
    float32_t b[len];
    float32_t c = 0;
    float32_t result;
    for (int32_t i = 0; i < len; i++) {
        a[i] = i;
        b[i] = i;
    }


    uint32_t time;
    float32_t gflops;
    time = benchmark(dotp_reference, iterations, &result, a, b, &c, len);
    gflops = ((iterations * len * 2 * 1000.0f)  / time) / 1000000000.0f;
    printf("C Reference (%d): %f, %f, %f\r\n", time, c, result, gflops);
    c = 0;
    RTC_Sleep(500);

    time = benchmark(dotp_scalar_intrinsic, iterations, &result, a, b, &c, len);
    gflops = ((iterations * len * 2 * 1000.0f)  / time) / 1000000000;
    printf("Scalar Intrinsic (%d): %f, %f, %f\r\n", time, c, result, gflops);
    c = 0;    
    RTC_Sleep(500);

    time = benchmark(dotp_mve_intrinsic, iterations, &result, a, b, &c, len);
    gflops = ((iterations * len * 2 * 1000.0f)  / time) / 1000000000.0f;
    printf("MVE Intrinsic (%d): %f, %f, %f\r\n", time, c, result, gflops);
    c = 0;
    RTC_Sleep(500);

    time = benchmark(dotp_scalar, iterations, &result, a, b, &c, len);
    gflops = ((iterations * len * 2 * 1000.0f)  / time) / 1000000000.0f;
    printf("ASM Scalar (%d): %f, %f, %f\r\n", time, c, result, gflops);

    c = 0;
    RTC_Sleep(500);
    time = benchmark(dotp_mve, iterations, &result, a, b, &c,  len);
    gflops = ((iterations * len * 2 * 1000.0f)  / time) / 1000000000.0f;
    printf("ASM MVE (%d): %f, %f, %f\r\n", time, c, result, gflops);

    stopTests();

    while (1) __WFE();
}

/*
#define TRAP_RET_ZERO  {__BKPT(0); return 0;}
int _close(int val) TRAP_RET_ZERO
int _lseek(int val0, int val1, int val2) TRAP_RET_ZERO
int _read(int val0, char * val1, int val2) TRAP_RET_ZERO
int _write(int val0, char * val1, int val2) TRAP_RET_ZERO
*/