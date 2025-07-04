#include <chrono>
#include <cstdint>
#include <ctime>
#include "RTE_Components.h"
#include "benchmark.hpp"
#include "profiling.hpp"
#include CMSIS_device_header
#include "SEGGER_RTT.h"
#include <cstdio>
#include "timing.hpp"
#include "LPRTC.hpp"
#include "fault_handler.h"


static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

extern "C" {
    void generateFpStall(float * input);
    void testPredication(float * input);
    void testDualIssue(float * a);
    void branch1(uint32_t count);
    void branch2(uint32_t count);
    void branchLOB(uint32_t count);
    void branchCBZ(uint32_t count);
    void testHeliumAlignment(uint32_t count, float const * arr);
    void testHeliumAlignmentNonAligned(uint32_t count, float const * arr);
    void testHeliumAlignmentLEUnaligned(uint32_t count, float const * arr);
}

void testBranching() {
    uint32_t time;
    uint32_t iterations = 100000000;
    RTC_Clock::time_point start;
    RTC_Clock::time_point end;

    SEGGER_RTT_printf(0, "Test;TimeAbs;Count\n");

    start = RTC_Clock::now();
    branch1(iterations);
    end = RTC_Clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "BranchBackward;%d;%d\r\n", time, iterations);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    
    start = RTC_Clock::now();
    branch2(iterations);
    end = RTC_Clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "BranchForward;%d;%d\r\n", time, iterations);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    start = RTC_Clock::now();
    branchLOB(iterations);
    end = RTC_Clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "BranchLOB;%d;%d\r\n", time, iterations);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    start = RTC_Clock::now();
    branchCBZ(iterations);
    end = RTC_Clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "BranchCBZ;%d;%d\r\n", time, iterations);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

}

void testAlignment() {
    float hallo[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    
    enableCpuClock();
    uint32_t count = 2 * pow(10, 7);
    auto start = CYCCNT_Clock::now();
    testHeliumAlignment(count, hallo);
    auto end = CYCCNT_Clock::now();
    uint32_t time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "Helium Aligned;%d\r\n", time);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    start = CYCCNT_Clock::now();
    testHeliumAlignmentNonAligned(count, hallo);
    end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "Helium Non-Aligned;%d\r\n", time);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

    start = CYCCNT_Clock::now();
    testHeliumAlignmentLEUnaligned(count, hallo);
    end = CYCCNT_Clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    sprintf(PRINTF_OUT_STRING, "Helium LE Non-Aligned;%d\r\n", time);
    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
}

int main (void) {
   	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

    float hallo[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // setupProfilingMVEStalls();
    // startCounting();
    // generateFpStall(hallo);
    // stopCounting();
    // printCounter();

    // testPredication(hallo);

    // setupProfilingDualIssue();
    // startCounting();
    // testDualIssue(hallo);
    // stopCounting();
    // printCounterDualIssue();

    testAlignment();

    while (1) __WFI();
}