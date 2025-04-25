#include <cstdio>
#include <cmath>
#include <cstdint>
#include "LPRTC.hpp"
#include "board.h"
#include "generators/Triad.hpp"
#include "fault_handler.h"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"

#include "generators/Simple.hpp"

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

__NO_RETURN int main() {
 	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

	using jitTest = void(*)(uint32_t *);
	uint32_t val = 3;

	// Thumb-Instruktionen sind Half-Word Aligned, d.h. bei 32Bit Instruktionen, müssen diese einfach in zwei "Instruktionen" aufgeteilt werden
	// Die CPU weiß dann, was damit zu tun ist
	// https://developer.arm.com/documentation/dui0473/m/overview-of-the-arm-architecture/arm-and-thumb-instruction-set-overview
	static const uint16_t instr[] = {
		0x6801, // LDR R1, [R0]
		0xf101, // ADD.W R1, R1, #3
		0x0103, // ADD.W R1, R1, #3
		0x6001, // STR R1, [R0]
		0x4770 // BX LR
	};
	__DSB(); // Data Synchronization Barrier
	__ISB(); // Instruction Synchronization Barrier

	jitTest jt = reinterpret_cast<jitTest>(reinterpret_cast<uintptr_t>(instr) | 1U); // LSB setzen für THUMB-Mode
	jt(&val);


	JIT::Generators::Simple simpleGen;
	JIT::Generators::Simple::Func simpleFunc = simpleGen.generate();
	uint32_t ret = simpleFunc();
	SEGGER_RTT_printf(0, "ret simple %d\n", ret);

	JIT::Generators::Triad triadGen;
	constexpr uint16_t count = 32;
	JIT::Generators::Triad::Func triadFunc = triadGen.generate(count);
	float a[count] = {0};
	float b[count] = {0};
	float c[count] = {0};
	for (uint32_t i = 0; i < count; i++) {
		a[i] = i;
		b[i] = i;
	}
	float scalar = 3.0;
	triadFunc(a, b, c, scalar);
	
	SEGGER_RTT_printf(0, "Val New %d\n", val);

	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}
}
