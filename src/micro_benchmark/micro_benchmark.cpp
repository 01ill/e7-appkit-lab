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
static constexpr uint32_t STREAM_ARRAY_SIZE = 48000;
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

# define HLINE "-------------------------------------------------------------\n"

static float a[STREAM_ARRAY_SIZE];
// static float b[STREAM_ARRAY_SIZE];
// static float c[STREAM_ARRAY_SIZE];

static float a_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));
// static float b_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));
// static float c_SRAM0[STREAM_ARRAY_SIZE] __attribute__((used, section(".bss.array_region_sram0")));

void initArrays() {
	for (uint32_t i = 0; i < STREAM_ARRAY_SIZE; i++) {
		a[i] = i;
		// b[i] = i;
		// c[i] = i;
		a_SRAM0[i] = i;
		// b_SRAM0[i] = i;
		// c_SRAM0[i] = i;
	}
}

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

// static uint32_t avgtime_stream[STREAM_TEST_COUNT] = {0};
// static uint32_t maxtime_stream[STREAM_TEST_COUNT] = {0};
// static uint32_t mintime_stream[STREAM_TEST_COUNT] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};

// static constexpr const char* label_stream[STREAM_TEST_COUNT] = {
// 	"Copy	",
// 	"Copy ASM",
// 	"Copy MVE",
// 	"Scale	",
// 	"Scale ASM",
// 	"Scale MVE",
//     "Add	",
// 	"Add ASM	",
//     "Add MVE	",
// 	"Triad	",
// 	"Triad ASM",
//     "Triad MVE",
// };

// static constexpr uint32_t bytes[STREAM_TEST_COUNT] = {
//     2 * sizeof(float) * STREAM_ARRAY_SIZE, // Copy braucht a,b = 2 * array
//     2 * sizeof(float) * STREAM_ARRAY_SIZE,
//     2 * sizeof(float) * STREAM_ARRAY_SIZE,
//     2 * sizeof(float) * STREAM_ARRAY_SIZE, // Scale braucht b,c = 2 * array
//     2 * sizeof(float) * STREAM_ARRAY_SIZE,
//     2 * sizeof(float) * STREAM_ARRAY_SIZE,
//     3 * sizeof(float) * STREAM_ARRAY_SIZE, // Add braucht a,b,c = 3 * array
// 	3 * sizeof(float) * STREAM_ARRAY_SIZE,
//     3 * sizeof(float) * STREAM_ARRAY_SIZE,
//     3 * sizeof(float) * STREAM_ARRAY_SIZE, // Triad braucht a,b,c = 3 * array
//     3 * sizeof(float) * STREAM_ARRAY_SIZE,
//     3 * sizeof(float) * STREAM_ARRAY_SIZE,
// };

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
	void throughput_mve_read(float * a, uint32_t len);
	void throughput_mve_read2(float * a, float * b, uint32_t len);
	void throughput_mve_write(float * a, uint32_t len);
	void throughput_scalar_read(float * a, uint32_t len);
	void throughput_scalar_write(float * a, uint32_t len);
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
	uint32_t targetThroughput = (2 * peak * pow(10, 9)); // 2x because two data accesses per cycle are supported
	uint32_t loopCount = targetThroughput / bytes;
	// loopCount = 100;
	uint32_t time;
	double gflops;
	SEGGER_RTT_printf(0, "Test;Time (µs);GFLOPS\n");
	for (uint32_t j = 0; j < ITERATIONS; j++) {
		initArrays();
		throughput_mve_read(a, STREAM_ARRAY_SIZE);
		start = CYCCNT_Clock::now();
		for (uint32_t i = 0; i < loopCount; i++) throughput_mve_read(a, STREAM_ARRAY_SIZE);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "Read Throughput;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
		
		initArrays();
		throughput_mve_write(a, STREAM_ARRAY_SIZE);
		start = CYCCNT_Clock::now();
		for (uint32_t i = 0; i < loopCount; i++) throughput_mve_write(a, STREAM_ARRAY_SIZE);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Write Throughput;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_mve_write(a_SRAM0, STREAM_ARRAY_SIZE);
		start = CYCCNT_Clock::now();
		for (uint32_t i = 0; i < loopCount; i++) throughput_mve_write(a_SRAM0, STREAM_ARRAY_SIZE);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "Write Throughput SRAM 0;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_mve_read(a_SRAM0, STREAM_ARRAY_SIZE);
		start = CYCCNT_Clock::now();
		for (uint32_t i = 0; i < loopCount; i++) throughput_mve_read(a_SRAM0, STREAM_ARRAY_SIZE);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time/1000000.0f * pow(10, 9));
		sprintf(PRINTF_OUT_STRING, "Read Throughput SRAM 0;%d;%f\r\n", time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	}
}

uint32_t throughputSizes[] = {512, 536, 560, 588, 616, 644, 672, 704, 740, 772, 808, 848, 888, 928, 972, 1020, 1068, 1116, 1168, 1224, 1280, 1340, 1404, 1472, 1540, 1612, 1688, 1768, 1848, 1936, 2028, 2124, 2224, 2328, 2436, 2552, 2672, 2796, 2928, 3064, 3208, 3360, 3516, 3680, 3856, 4036, 4224, 4424, 4632, 4848, 5076, 5312, 5564, 5824, 6096, 6384, 6684, 6996, 7324, 7668, 8028, 8404, 8800, 9212, 9644, 10096, 10572, 11068, 11588, 12128, 12700, 13296, 13920, 14572, 15256, 15972, 16720, 17504, 18328, 19188, 20088, 21028, 22016, 23048, 24128, 25260, 26448, 27688, 28988, 30348, 31772, 33260, 34820, 36456, 38164, 39956, 41832, 43792, 45848, 47996};
uint16_t throughputSizesLen = 100;

void benchmarkThroughputDifferentSizes() {
	auto start = CYCCNT_Clock::now();
	auto end = CYCCNT_Clock::now();
	uint32_t bytes = sizeof(float) * STREAM_ARRAY_SIZE;
	uint32_t time, k;
	uint32_t len = 0;
	double gflops;

	for (uint32_t i = 0; i < throughputSizesLen; i++) {
		len = throughputSizes[i];
		bytes = sizeof(float) * len;
		uint32_t targetThroughput = (peak * pow(10, 9));
		uint32_t loopCount = (targetThroughput / bytes);
		
		initArrays();
		throughput_mve_read(a_SRAM0, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_mve_read(a_SRAM0, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Read Throughput SRAM 0;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_scalar_read(a_SRAM0, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_scalar_read(a_SRAM0, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Scalar Read Throughput SRAM 0;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_mve_read(a, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_mve_read(a, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Read Throughput;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_scalar_read(a, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_scalar_read(a, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Scalar Read Throughput;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
		
		initArrays();
		throughput_mve_write(a_SRAM0, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_mve_write(a_SRAM0, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Write Throughput SRAM 0;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_scalar_write(a_SRAM0, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_scalar_write(a_SRAM0, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Scalar Write Throughput SRAM 0;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_mve_write(a, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_mve_write(a, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Write Throughput;%d;%d;%f\r\n", len, time, gflops);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

		initArrays();
		throughput_scalar_write(a, len);
		start = CYCCNT_Clock::now();
		for (k = 0; k < loopCount; k++) throughput_scalar_write(a, len);
		end = CYCCNT_Clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		gflops = targetThroughput / (time * 1000.0f);
		sprintf(PRINTF_OUT_STRING, "Scalar Write Throughput;%d;%d;%f\r\n", len, time, gflops);
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
            .RBAR = ARM_MPU_RBAR(0x02000000UL, ARM_MPU_SH_NON, 0UL, 1UL, 1UL), // RW, NP, XN
            .RLAR = ARM_MPU_RLAR(0x023FFFFFUL, 2UL) // SRAM0 with cacheable attribute
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
	ARM_MPU_SetMemAttr(3UL, ARM_MPU_ATTR(
        ARM_MPU_ATTR_MEMORY_(0,0,0,0), 
        ARM_MPU_ATTR_MEMORY_(0,0,0,0)
    ));
    ARM_MPU_SetMemAttr(4UL, ARM_MPU_ATTR( // No Cache
		ARM_MPU_ATTR_NON_CACHEABLE,
		ARM_MPU_ATTR_NON_CACHEABLE
    ));

    
    // Load the regions from the table
    ARM_MPU_Load(0U, &mpu_table[0], sizeof(mpu_table)/sizeof(ARM_MPU_Region_t));
    
    // Enable MPU with default memory map for privileged access
    ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);
}

int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();
	// configureMPU();
	// SCB_InvalidateDCache();
    // SCB_EnableDCache();
	// SCB_DisableDCache();
	enableCpuClock();

    // benchmarkFlops();
	// benchmarkThroughput();
	benchmarkThroughputDifferentSizes();
	disableCpuClock();
	while (1) {
		__WFE();
	}

	// /* Execute the STREAM Benchmark multiple times */
	// uint32_t times_stream[STREAM_TEST_COUNT][ITERATIONS];
	// float scalar = 3.0f;
    // for (uint32_t k = 0; k < ITERATIONS; k++) {
	// 	/* -- COPY --*/
	// 	times_stream[0][k] = LPRTC::getInstance().getCurrentValue();
	// 	for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
	// 		c[j] = a[j];
	// 	}
	// 	times_stream[0][k] = LPRTC::getInstance().getCurrentValue() - times_stream[0][k];

	// 	times_stream[1][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_copy(a, c, STREAM_ARRAY_SIZE);
	// 	times_stream[1][k] = LPRTC::getInstance().getCurrentValue() - times_stream[1][k];

	// 	times_stream[2][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_copy_mve(a, c, STREAM_ARRAY_SIZE);
	// 	times_stream[2][k] = LPRTC::getInstance().getCurrentValue() - times_stream[2][k];

	// 	/* -- SCALE --*/
	// 	times_stream[3][k] = LPRTC::getInstance().getCurrentValue();
	// 	for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
	// 		b[j] = scalar * c[j];
	// 	}
	// 	times_stream[3][k] = LPRTC::getInstance().getCurrentValue() - times_stream[3][k];

	// 	times_stream[4][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_scale(c, b, scalar, STREAM_ARRAY_SIZE);
	// 	times_stream[4][k] = LPRTC::getInstance().getCurrentValue() - times_stream[4][k];
		
	// 	times_stream[5][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_scale_mve(c, b, scalar, STREAM_ARRAY_SIZE);
	// 	times_stream[5][k] = LPRTC::getInstance().getCurrentValue() - times_stream[5][k];

	// 	/* -- ADD --*/
	// 	times_stream[6][k] = LPRTC::getInstance().getCurrentValue();
	// 	for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
	// 		c[j] = a[j] + b[j];
	// 	}
	// 	times_stream[6][k] = LPRTC::getInstance().getCurrentValue() - times_stream[6][k];
		
	// 	times_stream[7][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_add(c, a, b, STREAM_ARRAY_SIZE);
	// 	times_stream[7][k] = LPRTC::getInstance().getCurrentValue() - times_stream[7][k];

	// 	times_stream[8][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_add_mve(c, a, b, STREAM_ARRAY_SIZE);
	// 	times_stream[8][k] = LPRTC::getInstance().getCurrentValue() - times_stream[8][k];

	// 	/* -- TRIAD --*/
	// 	times_stream[9][k] = LPRTC::getInstance().getCurrentValue();
	// 	for (uint32_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
	// 		a[j] = b[j] + scalar * c[j];
	// 	}
	// 	times_stream[9][k] = LPRTC::getInstance().getCurrentValue() - times_stream[9][k];

	// 	times_stream[10][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_triad(a, b, c, scalar, STREAM_ARRAY_SIZE);
	// 	times_stream[10][k] = LPRTC::getInstance().getCurrentValue() - times_stream[10][k];

	// 	times_stream[11][k] = LPRTC::getInstance().getCurrentValue();
	// 	stream_triad_mve(a, b, c, scalar, STREAM_ARRAY_SIZE);
	// 	times_stream[11][k] = LPRTC::getInstance().getCurrentValue() - times_stream[11][k];
	// }

	// /* Calculate the STREAM results */
    // for (uint32_t k = 1; k < ITERATIONS; k++) {
	// 	for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
	// 		avgtime_stream[j] = avgtime_stream[j] + times_stream[j][k];
	// 		mintime_stream[j] = std::min(mintime_stream[j], times_stream[j][k]);
	// 		maxtime_stream[j] = std::max(maxtime_stream[j], times_stream[j][k]);
	// 	}
	// }

	// /* Print the STREAM results */
    // SEGGER_RTT_printf(0, "Function \t    Avg MB/s\t Avg time\t Best MB/s\t Min time\t Worst MB/s\t Max time\n");
    // for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
	// 	uint32_t avg = avgtime_stream[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
	// 	float avgSec = (float)avg / 32768.0f;
	// 	sprintf(PRINTF_OUT_STRING, "%s\t %12.1f\t %11.6f\t %12.1f\t %11.6f\t %12.1f\t %11.6f\n", label_stream[j],
	//     	1.0E-06 * bytes[j]/avgSec, avgSec, // Average
	// 	   	1.0E-06 * bytes[j]/(mintime_stream[j]/32768.0f), mintime_stream[j]/32768.0f, // Best
	// 	   	1.0E-06 * bytes[j]/(maxtime_stream[j]/32768.0f), maxtime_stream[j]/32768.0f // Worst
	// 	);
	//    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    // }
    // SEGGER_RTT_printf(0, HLINE);

	// /* Print the STREAM results as CSV */
	// SEGGER_RTT_printf(0, "Function ;    Avg MB/s; Avg time; Best MB/s; Min time; Worst MB/s; Max time\n");
	// for (uint32_t j = 0; j < STREAM_TEST_COUNT; j++) {
	// 	uint32_t avg = avgtime_stream[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
	// 	float avgSec = (float)avg / 32768.0f;
	// 	sprintf(PRINTF_OUT_STRING, "%s; %12.1f; %11.6f; %12.1f; %11.6f; %12.1f; %11.6f\n", label_stream[j],
	// 		1.0E-06 * bytes[j]/avgSec, avgSec, // Average
	// 		1.0E-06 * bytes[j]/(mintime_stream[j]/32768.0f), mintime_stream[j]/32768.0f, // Best
	// 		1.0E-06 * bytes[j]/(maxtime_stream[j]/32768.0f), maxtime_stream[j]/32768.0f // Worst
	// 	);
	// 	SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
	// }
	// SEGGER_RTT_printf(0, HLINE);
	
	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
