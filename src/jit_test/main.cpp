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
static constexpr uint32_t arrayMaxSize = 48;
static constexpr uint32_t M = 48;
static constexpr uint32_t K = 48;
static constexpr uint32_t N = 48;
static float bigA[M*K];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[K*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[M*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[M*N] __attribute__((used, section(".bss.array_region_sram0")));
// extern JIT::Instructions::Instruction16 globalBuffer[1024]; // __attribute__((section(".itcm_jit")));
JIT::Instructions::Instruction16 globalBuffer[1024] __attribute__((section(".itcm_jit"), aligned(4)));

__NO_RETURN int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();
/*
	JIT::Generators::PeakPerformance peakGen;
	for (uint32_t i = 0; i < peakCount; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	float scalar = 3.0;
	SEGGER_RTT_printf(0, "OperationalIntensity, GFLOPS\n");
	for (float opi = 0.25; opi < 1; opi += 0.25) {
		JIT::Generators::PeakPerformance::Func peakFunc = peakGen.generateSteps(opi);
		uint32_t start = LPRTC::getInstance().getCurrentValue();
		peakFunc(a, b, c, scalar, peakCount);
		uint32_t end = LPRTC::getInstance().getCurrentValue();
		// SEGGER_RTT_printf(0, "%d\n", end - start);

		float time = (float)(end - start) / 32768.0f;
		uint32_t flops = (opi * 8 * 4 * peakCount) / 4;
		float gflops = (flops / time) / pow(10, 9);
		printf("%f, %f\n", opi, gflops);
	}
	for (uint32_t opi = 1; opi < 120; opi++) {
		JIT::Generators::PeakPerformance::Func peakFunc = peakGen.generate(opi);
		uint32_t start = LPRTC::getInstance().getCurrentValue();
		peakFunc(a, b, c, scalar, peakCount);
		uint32_t end = LPRTC::getInstance().getCurrentValue();
		// SEGGER_RTT_printf(0, "%d\n", end - start);

		float time = (float)(end - start) / 32768.0f;
		uint32_t flops = (opi * 8 * 4 * peakCount) / 4;
		float gflops = (flops / time) / pow(10, 9);
		printf("%d, %f\n", opi, gflops);
	}
	SEGGER_RTT_printf(0, "OperationalIntensity, GFLOPS\n");
	for (float opi = 0.25; opi < 1; opi += 0.25) {
		JIT::Generators::PeakPerformance::Func peakFunc = peakGen.generateStepsNoMem(opi);
		uint32_t start = LPRTC::getInstance().getCurrentValue();
		peakFunc(a, b, c, scalar, peakCount);
		uint32_t end = LPRTC::getInstance().getCurrentValue();
		// SEGGER_RTT_printf(0, "%d\n", end - start);

		float time = (float)(end - start) / 32768.0f;
		uint32_t flops = (opi * 8 * 4 * peakCount) / 4;
		float gflops = (flops / time) / pow(10, 9);
		printf("%f, %f\n", opi, gflops);
	}
	for (uint32_t opi = 1; opi < 120; opi++) {
		JIT::Generators::PeakPerformance::Func peakFunc = peakGen.generateNoMem(opi);
		uint32_t start = LPRTC::getInstance().getCurrentValue();
		peakFunc(a, b, c, scalar, peakCount);
		uint32_t end = LPRTC::getInstance().getCurrentValue();
		// SEGGER_RTT_printf(0, "%d\n", end - start);

		float time = (float)(end - start) / 32768.0f;
		uint32_t flops = (opi * 8 * 4 * peakCount) / 4;
		float gflops = (flops / time) / pow(10, 9);
		printf("%d, %f\n", opi, gflops);
	}

	SEGGER_RTT_printf(0, "OperationalIntensity, VectorRegisters, GFLOPS\n");
	for (uint32_t vecCount = 2; vecCount <= 8; vecCount += 2) {
		for (uint32_t opi = 1; opi < 120; opi++) {
			JIT::Generators::PeakPerformance::Func peakFunc = peakGen.generate(opi, vecCount);
			uint32_t start = LPRTC::getInstance().getCurrentValue();
			peakFunc(a, b, c, scalar, peakCount);
			uint32_t end = LPRTC::getInstance().getCurrentValue();
			// SEGGER_RTT_printf(0, "%d\n", end - start);
	
			float time = (float)(end - start) / 32768.0f;
			uint32_t flops = (opi * 8 * 4 * peakCount) / 4;
			float gflops = (flops / time) / pow(10, 9);
			printf("%d, %d, %f\n", opi, vecCount, gflops);
		}
	}
*/
    for (uint32_t j = 0; j < arrayMaxSize*arrayMaxSize; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }

	JIT::Generators::Gemm gemmGen;
	JIT::Generators::Gemm::Func gemm24 = gemmGen.generate(M, K, N, M, K, N);
	gemm24 = gemmGen.bufferToFunc(globalBuffer);
	gemm24(bigA, bigB, bigC);

    uint32_t iterations = 8000;
    uint32_t time;
    float gflops;
    float result = 0.0f;

    uint32_t flops = iterations * 2 * M * K * N;

    RTC_Clock::time_point start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm24(bigA, bigB, bigC);
    }
    RTC_Clock::time_point end = RTC_Clock::now();

    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9));
    sprintf(PRINTF_OUT_STRING, "GEMM JIT %dx%dx%d (%d): %f, %f, %f\r\n", M, K, N, time, bigC[0], result, gflops);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

	for (uint32_t j = 0; j < M*K; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm24(bigA, bigB, bigC);
    gemm_reference_column_major(bigA, bigB, bigCRef, N, K, M, M, K, M);
    int32_t compareResult = compare(bigC, bigCRef, arrayMaxSize*arrayMaxSize);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "GEMM-JIT: Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "GEMM-JIT: Test nicht erfolgreich bei %d\n", compareResult);
    }



	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
