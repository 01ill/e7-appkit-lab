#include <arm_mve.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include "LPRTC.hpp"
#include "fault_handler.h"
#include "profiling.hpp"
#include "timing.hpp"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"

#ifdef M55_HE
constexpr float peak = 0.64;
#endif

#ifdef M55_HP
constexpr float peak = 1.6;
#endif


// #define USE_SRAM0

// should be minimum about 4 times of the available cache
// with 32KB of L1d Cache -> 8192floats
// but should also be big to ensure enough time between ticks
static constexpr uint32_t STREAM_ARRAY_SIZE = 10000;
static_assert(STREAM_ARRAY_SIZE % 4 == 0);
// count of tests
static constexpr uint32_t ITERATIONS = 3;

/**
the following STREAM benchmark tests are executed:
- Copy (C++, ASM Scalar, ASM MVE)
- Scale (C++, ASM Scalar, ASM MVE)
- Add (C++, ASM Scalar, ASM MVE)
- Triad (C++, ASM Scalar, ASM MVE)
*/
static constexpr uint8_t STREAM_TEST_COUNT = 12;
/**
the following tests are executed:
- Scalar FLOPS FP32
- Scalar FLOPS FP64
- Vector FLOPS FP16
- Vector FLOPS FP32
*/
static constexpr uint8_t FLOP_TEST_COUNT = 6;

# define HLINE "-------------------------------------------------------------\n"

static float a[STREAM_ARRAY_SIZE];
static float b[STREAM_ARRAY_SIZE];
static float c[STREAM_ARRAY_SIZE];

static float a_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));
static float b_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));
static float c_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

static uint32_t avgtime_stream[STREAM_TEST_COUNT] = {0};
static uint32_t maxtime_stream[STREAM_TEST_COUNT] = {0};
static uint32_t mintime_stream[STREAM_TEST_COUNT] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
static uint32_t avgtime_flops[FLOP_TEST_COUNT] = {0};
static uint32_t maxtime_flops[FLOP_TEST_COUNT] = {0};
static uint32_t mintime_flops[FLOP_TEST_COUNT] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};

static constexpr const char* label_stream[STREAM_TEST_COUNT] = {
	"Copy	",
	"Copy ASM",
	"Copy MVE",
	"Scale	",
	"Scale ASM",
	"Scale MVE",
    "Add	",
	"Add ASM	",
    "Add MVE	",
	"Triad	",
	"Triad ASM",
    "Triad MVE",
};
static constexpr const char* label_flops[FLOP_TEST_COUNT] = {
	"Scalar FP32",
	"Scalar FP64",
	"MVE FP16",
	"MVE FP32",
	"MVE FP32V4",
	"MVE FP32IL",
};

static constexpr uint32_t bytes[STREAM_TEST_COUNT] = {
    2 * sizeof(float) * STREAM_ARRAY_SIZE, // Copy braucht a,b = 2 * array
    2 * sizeof(float) * STREAM_ARRAY_SIZE,
    2 * sizeof(float) * STREAM_ARRAY_SIZE,
    2 * sizeof(float) * STREAM_ARRAY_SIZE, // Scale braucht b,c = 2 * array
    2 * sizeof(float) * STREAM_ARRAY_SIZE,
    2 * sizeof(float) * STREAM_ARRAY_SIZE,
    3 * sizeof(float) * STREAM_ARRAY_SIZE, // Add braucht a,b,c = 3 * array
	3 * sizeof(float) * STREAM_ARRAY_SIZE,
    3 * sizeof(float) * STREAM_ARRAY_SIZE,
    3 * sizeof(float) * STREAM_ARRAY_SIZE, // Triad braucht a,b,c = 3 * array
    3 * sizeof(float) * STREAM_ARRAY_SIZE,
    3 * sizeof(float) * STREAM_ARRAY_SIZE,
};

static constexpr uint32_t FLOP_MULTIPLIER = 15;
static constexpr uint32_t flop[FLOP_TEST_COUNT] = {
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 2 * 4 * STREAM_ARRAY_SIZE / 4, // FP32 FLOPS Scalar
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 2 * 4 * STREAM_ARRAY_SIZE / 4, // FP64 FLOPS Scalar
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 16 * 4 * STREAM_ARRAY_SIZE / 8, // FP16 FLOPS MVE
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 8 * 4 * STREAM_ARRAY_SIZE / 4, // FP32 FLOPS MVE
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 8 * 4 * STREAM_ARRAY_SIZE / 4, // FP32 FLOPS MVE
	FLOP_MULTIPLIER * OPERATIONAL_INTENSITY * 8 * 4 * STREAM_ARRAY_SIZE / 4 // FP32 FLOPS MVE
};

extern "C" {
	void stream_copy(float * __restrict a, float * __restrict c, uint32_t len);
	void stream_copy_mve(float * __restrict a, float * __restrict c, uint32_t len);
	void stream_scale(float * __restrict c, float * __restrict b, float scalar, uint32_t len);
	void stream_scale_mve(float * __restrict c, float * __restrict b, float scalar, uint32_t len);
	void stream_add(float * __restrict c, float * __restrict a, float * __restrict b, uint32_t len);
	void stream_add_mve(float * __restrict c, float * __restrict a, float * __restrict b, uint32_t len);
	void stream_triad(float * __restrict a, float * __restrict b, float * __restrict c, float scalar, uint32_t len);
	void stream_triad_mve(float * __restrict a, float * __restrict b, float * __restrict c, float scalar, uint32_t len);
	
	void flops_scalar_fp32(uint32_t len);
	void flops_scalar_fp64(uint32_t len);
	void flops_mve_fp16(uint32_t len);
	void flops_mve_fp32(uint32_t len);
	void flops_mve_fp32_interleaved(float * __restrict a, float * __restrict b, float * __restrict c, float scalar, uint32_t len);
	void flops_mve_fp32_vec4(uint32_t len);
	void throughput_mve(float * __restrict a, uint32_t len);
}

void benchmarkFlops() {
	uint32_t loopCount = (peak * pow(10, 9)) / (8 * 4);
	auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t time;
	double gflops;

	SEGGER_RTT_printf(0, "Test;Time (µs);GFLOPS\n");

	for (uint32_t k = 0; k < ITERATIONS; k++) {
		// loopCount = (peak * pow(10, 9)) / 4;
		start = CYCCNT_Clock::now();
		flops_scalar_fp32(loopCount / 2); // est. only half the performance
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = (static_cast<float>(peak * pow(10, 9))) / (time/1000000.0f * pow(10, 9) * 2);
		sprintf(PRINTF_OUT_STRING, "FLOPS Scalar FP32;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		start = CYCCNT_Clock::now();
		flops_scalar_fp64(loopCount / (2 * 23)); // est. only 1/23 of the performance of scalar fp32
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = static_cast<float>(peak * pow(10, 9)) / (time/1000000.0f * pow(10, 9) * 46);
		sprintf(PRINTF_OUT_STRING, "FLOPS Scalar FP64;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		start = CYCCNT_Clock::now();
		flops_mve_fp16(loopCount * 2); // est. double the performance
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = static_cast<float>(peak * pow(10, 9) * 2) / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "FLOPS MVE FP16;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


		start = CYCCNT_Clock::now();
		flops_mve_fp32(loopCount);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = static_cast<float>(peak * pow(10, 9)) / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "FLOPS MVE FP32;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	}
}

void benchmarkStream() {

}


void benchmarkThroughput() {
	auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t bytes = sizeof(float) * STREAM_ARRAY_SIZE;
	uint32_t time;
	double gflops;
	start = CYCCNT_Clock::now();
	throughput_mve(a, STREAM_ARRAY_SIZE);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(bytes) / (time/1000000.0f * pow(10, 9));
	sprintf(PRINTF_OUT_STRING, "Read Throughput;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	
	start = CYCCNT_Clock::now();
	throughput_mve(a_SRAM0, STREAM_ARRAY_SIZE);
	end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	gflops = static_cast<float>(bytes) / (time/1000000.0f * pow(10, 9));
	sprintf(PRINTF_OUT_STRING, "Read Throughput SRAM 0;%d;%f\r\n", time, gflops);
	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

}

void benchmarkThroughputDifferentSizes() {
	auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t bytes = sizeof(float) * STREAM_ARRAY_SIZE;
	uint32_t time;
	double gflops;


	for (uint32_t s = 256; s < STREAM_ARRAY_SIZE; s += 1000) {
		bytes = sizeof(float) * s * 10000;
		start = CYCCNT_Clock::now();
		for (uint32_t k = 0; k < 10000; k++) throughput_mve(a_SRAM0, s);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = static_cast<float>(bytes) / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "Read Throughput SRAM 0;%d;%d;%f\r\n", s, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	}

}

 int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();
	enableCpuClock();

    SEGGER_RTT_printf(0, HLINE);
    benchmarkFlops();
	// benchmarkThroughput();
	disableCpuClock();
	// while (1) {
	// 	__WFE();
	// }

	/* Execute the STREAM Benchmark multiple times */
	uint32_t times_stream[STREAM_TEST_COUNT][ITERATIONS];
	float scalar = 3.0f;
    for (uint32_t k = 0; k < ITERATIONS; k++) {
		/* -- COPY --*/
		times_stream[0][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
			c[j] = a[j];
		}
		times_stream[0][k] = LPRTC::getInstance().getCurrentValue() - times_stream[0][k];

		times_stream[1][k] = LPRTC::getInstance().getCurrentValue();
		stream_copy(a, c, STREAM_ARRAY_SIZE);
		times_stream[1][k] = LPRTC::getInstance().getCurrentValue() - times_stream[1][k];

		times_stream[2][k] = LPRTC::getInstance().getCurrentValue();
		stream_copy_mve(a, c, STREAM_ARRAY_SIZE);
		times_stream[2][k] = LPRTC::getInstance().getCurrentValue() - times_stream[2][k];

		/* -- SCALE --*/
		times_stream[3][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
			b[j] = scalar * c[j];
		}
		times_stream[3][k] = LPRTC::getInstance().getCurrentValue() - times_stream[3][k];

		times_stream[4][k] = LPRTC::getInstance().getCurrentValue();
		stream_scale(c, b, scalar, STREAM_ARRAY_SIZE);
		times_stream[4][k] = LPRTC::getInstance().getCurrentValue() - times_stream[4][k];
		
		times_stream[5][k] = LPRTC::getInstance().getCurrentValue();
		stream_scale_mve(c, b, scalar, STREAM_ARRAY_SIZE);
		times_stream[5][k] = LPRTC::getInstance().getCurrentValue() - times_stream[5][k];

		/* -- ADD --*/
		times_stream[6][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
			c[j] = a[j] + b[j];
		}
		times_stream[6][k] = LPRTC::getInstance().getCurrentValue() - times_stream[6][k];
		
		times_stream[7][k] = LPRTC::getInstance().getCurrentValue();
		stream_add(c, a, b, STREAM_ARRAY_SIZE);
		times_stream[7][k] = LPRTC::getInstance().getCurrentValue() - times_stream[7][k];

		times_stream[8][k] = LPRTC::getInstance().getCurrentValue();
		stream_add_mve(c, a, b, STREAM_ARRAY_SIZE);
		times_stream[8][k] = LPRTC::getInstance().getCurrentValue() - times_stream[8][k];

		/* -- TRIAD --*/
		times_stream[9][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
			a[j] = b[j] + scalar * c[j];
		}
		times_stream[9][k] = LPRTC::getInstance().getCurrentValue() - times_stream[9][k];

		times_stream[10][k] = LPRTC::getInstance().getCurrentValue();
		stream_triad(a, b, c, scalar, STREAM_ARRAY_SIZE);
		times_stream[10][k] = LPRTC::getInstance().getCurrentValue() - times_stream[10][k];

		times_stream[11][k] = LPRTC::getInstance().getCurrentValue();
		stream_triad_mve(a, b, c, scalar, STREAM_ARRAY_SIZE);
		times_stream[11][k] = LPRTC::getInstance().getCurrentValue() - times_stream[11][k];
	}

	/* Calculate the STREAM results */
    for (uint32_t k = 1; k < ITERATIONS; k++) {
		for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
			avgtime_stream[j] = avgtime_stream[j] + times_stream[j][k];
			mintime_stream[j] = std::min(mintime_stream[j], times_stream[j][k]);
			maxtime_stream[j] = std::max(maxtime_stream[j], times_stream[j][k]);
		}
	}

	/* Print the STREAM results */
    SEGGER_RTT_printf(0, "Function \t    Avg MB/s\t Avg time\t Best MB/s\t Min time\t Worst MB/s\t Max time\n");
    for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
		uint32_t avg = avgtime_stream[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
		float avgSec = (float)avg / 32768.0f;
		sprintf(PRINTF_OUT_STRING, "%s\t %12.1f\t %11.6f\t %12.1f\t %11.6f\t %12.1f\t %11.6f\n", label_stream[j],
	    	1.0E-06 * bytes[j]/avgSec, avgSec, // Average
		   	1.0E-06 * bytes[j]/(mintime_stream[j]/32768.0f), mintime_stream[j]/32768.0f, // Best
		   	1.0E-06 * bytes[j]/(maxtime_stream[j]/32768.0f), maxtime_stream[j]/32768.0f // Worst
		);
	   SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    }
    SEGGER_RTT_printf(0, HLINE);

	/* Print the STREAM results as CSV */
	SEGGER_RTT_printf(0, "Function ;    Avg MB/s; Avg time; Best MB/s; Min time; Worst MB/s; Max time\n");
	for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
		uint32_t avg = avgtime_stream[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
		float avgSec = (float)avg / 32768.0f;
		sprintf(PRINTF_OUT_STRING, "%s; %12.1f; %11.6f; %12.1f; %11.6f; %12.1f; %11.6f\n", label_stream[j],
			1.0E-06 * bytes[j]/avgSec, avgSec, // Average
			1.0E-06 * bytes[j]/(mintime_stream[j]/32768.0f), mintime_stream[j]/32768.0f, // Best
			1.0E-06 * bytes[j]/(maxtime_stream[j]/32768.0f), maxtime_stream[j]/32768.0f // Worst
		);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	}
	SEGGER_RTT_printf(0, HLINE);
	
	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
