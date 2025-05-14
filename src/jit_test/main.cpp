#include <cstdio>
#include <cmath>
#include <cstdint>
#include "LPRTC.hpp"
#include "board.h"
#include "generators/PeakPerformance.hpp"
#include "generators/Triad.hpp"
#include "generators/Gemm.hpp"
#include "fault_handler.h"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"
#include "benchmark.hpp"
#include "timing.hpp"
#include "arm_math.h"

#include "generators/Simple.hpp"

float gemm_reference_column_major(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, const uint32_t n, const uint32_t k, const uint32_t m, const uint32_t lda, const uint32_t ldb, const uint32_t ldc) {
    for (uint32_t j = 0; j < n; j++) { // j = n
        for (uint32_t i = 0; i < m; i++) { // i = m
            // c[j * len + i] = 0.0f;
            for (uint32_t p = 0; p < k; p++) { // p = k
                c[j * ldc + i] += a[p * lda + i] * b[j * ldb + p];
            }
        }
    }
    return c[0];
}



static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));
/*constexpr uint32_t peakCount = 50000;
static float a[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float b[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float c[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
*/
static constexpr uint32_t arrayMaxSize = 256;
static uint32_t M = 27;
static uint32_t K = 24;
static uint32_t N = 26;
static float bigA[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
// extern JIT::Instructions::Instruction16 globalBuffer[1024]; // __attribute__((section(".itcm_jit")));
JIT::Instructions::Instruction16 globalBuffer[1024] __attribute__((section(".itcm_jit"), aligned(4)));

void initMatrices(float * a, float * b, float * c, float * cref, const uint32_t m, const uint32_t n, const uint32_t k) {
    for (uint32_t i = 0; i < m*k; i++) a[i] = i;
    for (uint32_t i = 0; i < k*n; i++) b[i] = i;
    for (uint32_t i = 0; i < m*n; i++) c[i] = 0;
	for (uint32_t i = 0; i < m*n; i++) cref[i] = 0;
}


__NO_RETURN int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

	initMatrices(bigA, bigB, bigC, bigCRef, M, N, K);
	JIT::Generators::Gemm gemmGen;
	JIT::Generators::Gemm::Func gemm24 = gemmGen.generate(M, K, N, M, K, M);
	gemm24 = gemmGen.bufferToFunc(globalBuffer);
	// gemm24(bigA, bigB, bigC);
    uint32_t iterations = 8000;
    uint32_t time;
    float gflops;
    float result = 0.0f;
    arm_matrix_instance_f32 armA;
    arm_matrix_instance_f32 armB;
    arm_matrix_instance_f32 armC;
    arm_status status;

    uint32_t flops = iterations * 2 * M * K * N;
    RTC_Clock::time_point start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm24(bigA, bigB, bigC);
    }
    RTC_Clock::time_point end = RTC_Clock::now();

    arm_mat_init_f32(&armA, M, K, bigA);
    arm_mat_init_f32(&armB, K, N, bigB);
    arm_mat_init_f32(&armC, M, N, bigC);

    time = benchmarkArm(arm_mat_mult_f32, iterations, &status, &armA, &armB, &armC);
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "CMSIS-DSP %dx%dx%d (%d): %f, %f, %f\r\n", M, K, N, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "GEMM JIT %dx%dx%d (%d): %f, %f, %f\r\n", M, K, N, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

	initMatrices(bigA, bigB, bigC, bigCRef, M, N, K);
    gemm24(bigA, bigB, bigC);
    gemm_reference_column_major(bigA, bigB, bigCRef, N, K, M, M, K, M);
    int32_t compareResult = compare(bigC, bigCRef, M*N);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "GEMM-JIT: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "GEMM-JIT: Test nicht erfolgreich bei %d\n", compareResult);
    }

    SEGGER_RTT_printf(0, "M;K;N;Type;GFLOPS;Correct\n");
    for (uint32_t i = 8; i < 60; i++) {
        //M = RTC_GetTimepoint() % 50 + 10;
        //RTC_Sleep(M);
        //N = RTC_GetTimepoint() % 50 + 10;
        //RTC_Sleep(N);
        //K = RTC_GetTimepoint() % 50 + 10;
        M = i;
        N = i;
        K = 5*i;
        uint32_t flops = iterations * 2 * M * K * N;

        arm_mat_init_f32(&armA, M, K, bigA);
        arm_mat_init_f32(&armB, K, N, bigB);
        arm_mat_init_f32(&armC, M, N, bigC);

        time = benchmarkArm(arm_mat_mult_f32, iterations, &status, &armA, &armB, &armC);
        gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
        sprintf(PRINTF_OUT_STRING, "%d;%d;%d;CMSIS-DSP;%f;1\r\n", M, K, N, gflops);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

        gemm24 = gemmGen.generate(M, K, N, M, K, M);
	    gemm24 = gemmGen.bufferToFunc(globalBuffer);
        start = RTC_Clock::now();
        for (uint32_t j = 0; j < iterations; j++) {
            gemm24(bigA, bigB, bigC);
        }
        end = RTC_Clock::now();

        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
        initMatrices(bigA, bigB, bigC, bigCRef, M, N, K);
        gemm24(bigA, bigB, bigC);
        gemm_reference_column_major(bigA, bigB, bigCRef, N, K, M, M, K, M);
        int32_t compareResult = compare(bigC, bigCRef, M*N);
        if (compareResult == -1) {
            sprintf(PRINTF_OUT_STRING, "%d;%d;%d;JIT;%f;1\r\n", M, K, N, gflops);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        } else {
            sprintf(PRINTF_OUT_STRING, "%d;%d;%d;JIT;%f;0\r\n", M, K, N, gflops);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }

	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
