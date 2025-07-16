#include "jit_tests.hpp"
#include <cstdint>
#include "instructions/Base.hpp"
#include "timing.hpp"
#include "SEGGER_RTT.h"
#include "../generators/PeakPerformance.hpp"
#include "../generators/Throughput.hpp"

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

void testPeakPerformance(JIT::Instructions::Instruction16 * globalBuffer, JIT::Instructions::Instruction16 * globalBufferSram0, JIT::Instructions::Instruction16 * globalBufferDtcm, uint32_t arrayMaxSize) {
    uint32_t oi = 1;
    uint32_t iterations = 100;
    auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t flops = (oi * 8 * 4 * arrayMaxSize * 10000);
	uint32_t time;
	double gflops;
    JIT::Generators::PeakPerformance gen(globalBuffer, 10000);
    JIT::Generators::PeakPerformance::Func genFunc = gen.generate(oi);
    // genFunc = gen.bufferToFunc(globalBuffer);
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(arrayMaxSize * 10000);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "PeakJIT ITCM;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    JIT::Generators::PeakPerformance gen2(globalBufferSram0, 100000);
    genFunc = gen2.generate(oi);
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(arrayMaxSize * 10000);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "PeakJIT SRAM0;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    JIT::Generators::PeakPerformance gen3(globalBufferDtcm, 10000);
    genFunc = gen3.generate(oi);
    // genFunc = gen.bufferToFunc(globalBuffer);
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(arrayMaxSize * 10000);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "PeakJIT DTCM;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
}

void testThroughput(JIT::Instructions::Instruction16 * globalBuffer, JIT::Instructions::Instruction16 * globalBufferSram0, JIT::Instructions::Instruction16 * globalBufferDtcm, uint32_t arrayMaxSize, float * bigA) {
    uint32_t iterations = 10000;
    auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t flops = (4 * arrayMaxSize * arrayMaxSize);
	uint32_t time;
	double gflops;
    JIT::Generators::Throughput gen(globalBuffer, 3072);
    JIT::Generators::Throughput::Func genFunc = gen.generate();
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(bigA, arrayMaxSize*arrayMaxSize);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "Throughput JIT ITCM;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    JIT::Generators::Throughput genDtcm(globalBufferDtcm, 3072);
    genFunc = genDtcm.generate();
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(bigA, arrayMaxSize*arrayMaxSize);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "Throughput JIT DTCM;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    JIT::Generators::Throughput genSram0(globalBufferSram0, 3072);
    genFunc = genSram0.generate();
	start = CYCCNT_Clock::now();
    for (uint32_t i = 0; i < iterations; i++) genFunc(bigA, arrayMaxSize*arrayMaxSize);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(flops) / (time/1000000.0f * pow(10, 9)) * iterations;
	sprintf(PRINTF_OUT_STRING, "Throughput JIT SRAM0;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
}
