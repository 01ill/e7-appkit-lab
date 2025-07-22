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

#include "helper/jit_tests.hpp"
#include "helper/gemm_kernel.hpp"
#include "helper/gemm_tests.hpp"

#ifdef M55_HE
constexpr float peak = 0.64;
#endif

#ifdef M55_HP
constexpr float peak = 1.6;
#endif

constexpr bool testReference = true;
constexpr bool testIntrinsics = true;
constexpr bool testArm = true;
constexpr bool testJitter = true;

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
static constexpr uint32_t arrayMaxSize = 128;
static const uint32_t M = 1;
static const uint32_t K = 28;
static const uint32_t N = 9;
static float bigA[M*K];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[K*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[M*N];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[M*N];// __attribute__((used, section(".bss.array_region_sram0")));
#else
static constexpr uint32_t arrayMaxSize = 100;
static float bigA[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigB[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigC[arrayMaxSize*arrayMaxSize];// __attribute__((used, section(".bss.array_region_sram0")));
static float bigCRef[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float aSram0[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float bSram0[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float cSram0[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float cRefSram0[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));

#endif

JIT::Instructions::Instruction16 globalBuffer[8192] __attribute__((section(".itcm_jit"), aligned(4)));
JIT::Instructions::Instruction16 globalBufferDtcm[8192] __attribute__((aligned(4)));
JIT::Instructions::Instruction16 globalBufferSram0[8192] __attribute__((section(".sram0_jit"), aligned(4)));

// JIT::Instructions::Instruction16 globalBuffer[3072] __attribute__((aligned(4)));

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
    // enableCpuClock();
    // testThroughput(globalBuffer, globalBufferSram0, globalBufferDtcm, arrayMaxSize, bigA);
    // testPeakPerformance(globalBuffer, globalBufferSram0, globalBufferDtcm, arrayMaxSize);
    // disableCpuClock();
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

    constSizeTest(bigA, bigB, bigC, bigCRef, globalBuffer, m, n, k, false);

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
    // testSquareShapes(bigA, bigB, bigC, bigCRef, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    testGrowingK(bigA, bigB, bigC, bigCRef, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    testGrowingM(bigA, bigB, bigC, bigCRef, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    testGrowingN(bigA, bigB, bigC, bigCRef, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    // testSquareShapes(aSram0, bSram0, cSram0, cRefSram0, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    // testGrowingK(aSram0, bSram0, cSram0, cRefSram0, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    // testGrowingM(aSram0, bSram0, cSram0, cRefSram0, globalBuffer, testArm, testJitter, testIntrinsics, testReference);
    // testGrowingN(aSram0, bSram0, cSram0, cRefSram0, globalBuffer, testArm, testJitter, testIntrinsics, testReference);

    // testAllSizes(bigA, bigB, bigC, bigCRef, globalBuffer, testArm, testJitter, testIntrinsics, testReference, 1, 16, 13, false);
#endif
	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
