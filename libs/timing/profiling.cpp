#include "profiling.hpp"
#ifdef M55_HP
#include "M55_HP.h"
#endif
#ifdef M55_HE
#include "M55_HE.h"
#endif
#include "SEGGER_RTT.h"
#include <cstdint>
#include <cstdio>

void setupProfilingMVEStalls() {
    ARM_PMU_Enable();

	ARM_PMU_Set_EVTYPER(0, ARM_PMU_INST_RETIRED); // counter reg 0 count instructions retired
	// ARM_PMU_Set_EVTYPER(1, ARM_PMU_L1D_CACHE_MISS_RD); 
	ARM_PMU_Set_EVTYPER(2, ARM_PMU_MVE_STALL);
	ARM_PMU_Set_EVTYPER(3, ARM_PMU_MVE_STALL_DEPENDENCY);
	ARM_PMU_Set_EVTYPER(4, ARM_PMU_MVE_STALL_RESOURCE);
	ARM_PMU_Set_EVTYPER(5, ARM_PMU_MVE_STALL_RESOURCE_MEM);
	ARM_PMU_Set_EVTYPER(6, ARM_PMU_MVE_STALL_RESOURCE_FP);
	ARM_PMU_Set_EVTYPER(7, ARM_PMU_MVE_STALL_RESOURCE_INT);
	ARM_PMU_Set_EVTYPER(8, ARM_PMU_MVE_STALL_BREAK);
	ARM_PMU_Set_EVTYPER(6, ARM_PMU_MVE_INST_RETIRED);
	ARM_PMU_Set_EVTYPER(7, ARM_PMU_MVE_LDST_RETIRED);
}

void setupProfilingStalls() {
    ARM_PMU_Enable();

	ARM_PMU_Set_EVTYPER(0, ARM_PMU_STALL);
	ARM_PMU_Set_EVTYPER(1, ARM_PMU_STALL_BACKEND);
	ARM_PMU_Set_EVTYPER(2, ARM_PMU_STALL_FRONTEND);
	ARM_PMU_Set_EVTYPER(3, ARM_PMU_STALL_OP);
	ARM_PMU_Set_EVTYPER(4, ARM_PMU_STALL_OP_BACKEND);
	ARM_PMU_Set_EVTYPER(5, ARM_PMU_STALL_OP_FRONTEND);
}

void setupProfilingMemory() {
	ARM_PMU_Enable();

	ARM_PMU_Set_EVTYPER(0, ARM_PMU_L1D_CACHE);
	ARM_PMU_Set_EVTYPER(1, ARM_PMU_L1D_CACHE_MISS_RD); 
	ARM_PMU_Set_EVTYPER(2, ARM_PMU_L1D_CACHE_RD);
	ARM_PMU_Set_EVTYPER(3, ARM_PMU_L1I_CACHE);
}

void setupProfilingMVEInstructions() {
	ARM_PMU_Enable();

	ARM_PMU_Set_EVTYPER(0, ARM_PMU_CPU_CYCLES);
	ARM_PMU_Set_EVTYPER(1, ARM_PMU_INST_RETIRED);
	ARM_PMU_Set_EVTYPER(2, ARM_PMU_MVE_INST_RETIRED);
	ARM_PMU_Set_EVTYPER(3, ARM_PMU_MVE_STALL);
}

void setupProfilingDualIssue() {
	ARM_PMU_Enable();

	ARM_PMU_Set_EVTYPER(0, ARM_PMU_CPU_CYCLES);
	ARM_PMU_Set_EVTYPER(1, ARM_PMU_INST_RETIRED);
	ARM_PMU_Set_EVTYPER(2, ARM_PMU_OP_COMPLETE);
	ARM_PMU_Set_EVTYPER(3, ARM_PMU_STALL);
}

void startCounting() {
	// DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	// DWT->CTRL |= DWT_CTRL_FOLDEVTENA_Msk;
    ARM_PMU_EVCNTR_ALL_Reset();
	ARM_PMU_CYCCNT_Reset();
	ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk | PMU_CNTENSET_CNT0_ENABLE_Msk | PMU_CNTENSET_CNT1_ENABLE_Msk | PMU_CNTENSET_CNT2_ENABLE_Msk | PMU_CNTENSET_CNT3_ENABLE_Msk 
		| PMU_CNTENSET_CNT4_ENABLE_Msk | PMU_CNTENSET_CNT5_ENABLE_Msk | PMU_CNTENSET_CNT6_ENABLE_Msk | PMU_CNTENSET_CNT7_ENABLE_Msk);
}

void stopCounting() {
	ARM_PMU_CNTR_Disable(PMU_CNTENSET_CCNTR_ENABLE_Msk | PMU_CNTENSET_CNT0_ENABLE_Msk | PMU_CNTENSET_CNT1_ENABLE_Msk | PMU_CNTENSET_CNT2_ENABLE_Msk | PMU_CNTENSET_CNT3_ENABLE_Msk 
		| PMU_CNTENSET_CNT4_ENABLE_Msk | PMU_CNTENSET_CNT5_ENABLE_Msk | PMU_CNTENSET_CNT6_ENABLE_Msk | PMU_CNTENSET_CNT7_ENABLE_Msk);
	// ARM_PMU_CNTR_Increment(PMU_SWINC_CNT4_Msk);
}

void printCounter() {
    uint32_t cycle_count = ARM_PMU_Get_CCNTR();
	uint32_t instructions_retired_count = ARM_PMU_Get_EVCNTR(0);
	uint32_t l1_dcache_miss_count = ARM_PMU_Get_EVCNTR(1);
	uint32_t mve_stall = ARM_PMU_Get_EVCNTR(2);
	uint32_t mve_stall_dependency = ARM_PMU_Get_EVCNTR(3);
	uint32_t mve_stall_resource = ARM_PMU_Get_EVCNTR(4);
	uint32_t mve_stall_resource_mem = ARM_PMU_Get_EVCNTR(5);
	uint32_t mve_stall_resource_fp = ARM_PMU_Get_EVCNTR(6);
	uint32_t mve_stall_resource_int = ARM_PMU_Get_EVCNTR(7);
	uint32_t mve_stall_break = ARM_PMU_Get_EVCNTR(8);

    SEGGER_RTT_printf(0, "--- Write Counter ---\n");
    SEGGER_RTT_printf(0, "Cycle Count: %d\n", cycle_count);
    SEGGER_RTT_printf(0, "Instructions Retired: %d\n", instructions_retired_count);
    SEGGER_RTT_printf(0, "L1D Cache Misses: %d\n", l1_dcache_miss_count);
    SEGGER_RTT_printf(0, "MVE Stalls: %d\n", mve_stall);
    SEGGER_RTT_printf(0, "-- MVE Dependency Stalls: %d\n", mve_stall_dependency);
    SEGGER_RTT_printf(0, "-- MVE Resource Stalls: %d\n", mve_stall_resource);
    SEGGER_RTT_printf(0, "-- MVE Resource MEM Stalls: %d\n", mve_stall_resource_mem);
    SEGGER_RTT_printf(0, "-- MVE Resource FP Stalls: %d\n", mve_stall_resource_fp);
    SEGGER_RTT_printf(0, "-- MVE Resource INT Stalls: %d\n", mve_stall_resource_int);
    SEGGER_RTT_printf(0, "-- MVE Break Stalls: %d\n", mve_stall_break);
	SEGGER_RTT_printf(0, "Stalls: %d\n", ARM_PMU_Get_EVCNTR(1));
	SEGGER_RTT_printf(0, "-- Backend Stalls: %d\n", ARM_PMU_Get_EVCNTR(10));
	SEGGER_RTT_printf(0, "-- Frontend Stalls: %d\n", ARM_PMU_Get_EVCNTR(11));

	SEGGER_RTT_printf(0, "-- OP Stalls: %d\n", ARM_PMU_Get_EVCNTR(1));
	SEGGER_RTT_printf(0, "-- -- OP Backend Stalls: %d\n", ARM_PMU_Get_EVCNTR(13));
	SEGGER_RTT_printf(0, "-- -- OP Frontend Stalls: %d\n", ARM_PMU_Get_EVCNTR(14));
	SEGGER_RTT_printf(0, "MVE FP Instructions: %d\n", ARM_PMU_Get_EVCNTR(15));
	SEGGER_RTT_printf(0, "MVE LDST Instructions: %d\n", ARM_PMU_Get_EVCNTR(16));
	SEGGER_RTT_printf(0, "L1D Cache Access: %d\n", ARM_PMU_Get_EVCNTR(1));
}

void printCounterStalls() {
	SEGGER_RTT_printf(0, "Stalls: %d\n", ARM_PMU_Get_EVCNTR(0));
	SEGGER_RTT_printf(0, "-- Backend Stalls: %d\n", ARM_PMU_Get_EVCNTR(1));
	SEGGER_RTT_printf(0, "-- Frontend Stalls: %d\n", ARM_PMU_Get_EVCNTR(2));

	SEGGER_RTT_printf(0, "-- OP Stalls: %d\n", ARM_PMU_Get_EVCNTR(3));
	SEGGER_RTT_printf(0, "-- -- OP Backend Stalls: %d\n", ARM_PMU_Get_EVCNTR(4));
	SEGGER_RTT_printf(0, "-- -- OP Frontend Stalls: %d\n", ARM_PMU_Get_EVCNTR(5));

}

void printCounterMemory() {
	SEGGER_RTT_printf(0, "L1D: %d\n", ARM_PMU_Get_EVCNTR(0));
	SEGGER_RTT_printf(0, "L1D Miss: %d\n", ARM_PMU_Get_EVCNTR(1));
	SEGGER_RTT_printf(0, "L1D Read: %d\n", ARM_PMU_Get_EVCNTR(2));
	SEGGER_RTT_printf(0, "L1I: %d\n", ARM_PMU_Get_EVCNTR(3));
}

void printCounterMVEInstructions() {
	SEGGER_RTT_printf(0, "Cycles: %d\n", ARM_PMU_Get_CCNTR());
	SEGGER_RTT_printf(0, "Cycle Count: %d\n", ARM_PMU_Get_EVCNTR(0));
	SEGGER_RTT_printf(0, "Instruction Count: %d\n", ARM_PMU_Get_EVCNTR(1));
	SEGGER_RTT_printf(0, "MVE Instructions: %d\n", ARM_PMU_Get_EVCNTR(2));
	SEGGER_RTT_printf(0, "MVE Stall Cycles: %d\n", ARM_PMU_Get_EVCNTR(3));
	// see https://kannwischer.eu/papers/2022_ntt-int-mul.pdf, p. 19
}

void printCounterDualIssue() {
	uint32_t cycle_count = ARM_PMU_Get_CCNTR();
	uint32_t inst_retired = ARM_PMU_Get_EVCNTR(1);
	// float ipc = (float)inst_retired / cycle_count;
	SEGGER_RTT_printf(0, "Cycle Count: %d\n", cycle_count);
	// SEGGER_RTT_printf(0, "Cycle Count 2: %d\n", DWT->CYCCNT);
	// SEGGER_RTT_printf(0, "Fold: %d\n", DWT->FOLDCNT);
	SEGGER_RTT_printf(0, "Instruction Count: %d\n", inst_retired);


	// SEGGER_RTT_printf(0, "IPC: %d/100\n", ipc * 100);
	SEGGER_RTT_printf(0, "OP Complete: %d\n", ARM_PMU_Get_EVCNTR(2));
	SEGGER_RTT_printf(0, "Stall Cycles: %d\n", ARM_PMU_Get_EVCNTR(3));
}