#include <cstdio>
#include <cmath>
#include <cstdint>
#include "LPRTC.hpp"
#include "board.h"
#include "generators/PeakPerformance.hpp"
#include "generators/Triad.hpp"
#include "fault_handler.h"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"

#include "generators/Simple.hpp"

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));
constexpr uint32_t peakCount = 300000;
static float a[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float b[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));
static float c[peakCount];// __attribute__((used, section(".bss.array_region_sram0")));

__NO_RETURN int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

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


	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
