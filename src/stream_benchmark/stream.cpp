#include <arm_mve.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include "LPRTC.hpp"
#include "fault_handler.h"
#include <RTE_Components.h>
#include CMSIS_device_header
#include "SEGGER_RTT.h"


static constexpr uint32_t ARRAY_SIZE = 80000;
static constexpr uint32_t ITERATIONS = 100;
static constexpr uint32_t RESULTS = 17;

# define HLINE "-------------------------------------------------------------\n"


using STREAM_TYPE = float;
static STREAM_TYPE a[ARRAY_SIZE];//  __attribute__((used, section(".bss.array_region_sram0")));
static STREAM_TYPE b[ARRAY_SIZE];//  __attribute__((used, section(".bss.array_region_sram0")));
static STREAM_TYPE c[ARRAY_SIZE];//  __attribute__((used, section(".bss.array_region_sram0")));
static double fp64_array[32];
static float16_t fp16_array[32];

static uint32_t avgtime[RESULTS] = {0};
static uint32_t maxtime[RESULTS] = {0};
static uint32_t mintime[RESULTS] = {UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};

static constexpr const char* label[RESULTS] = {
	"Copy	",
	"Copy ASM",
	"Copy MVE",
	"Scale	",
	"Scale ASM",
	"Scale MVE",
    "Add	",
	"Add ASM",
    "Add MVE",
	"Triad	",
	"Triad ASM",
    "Triad MVE",
	"Triad FLOPS",
	"Scalar FP16",
	"Scalar FP32",
	"Scalar FP64",
	"MVE FP16"
};

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));

static constexpr uint32_t bytes[RESULTS] = {
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // Copy braucht a,b = 2 * array
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // Scale braucht b,c = 2 * array
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // Add braucht a,b,c = 3 * array
	3 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // Triad braucht a,b,c = 3 * array
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE,
	3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // FP16 FLOPS Scalar
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // FP32 FLOPS Scalar
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // FP64 FLOPS Scalar
    3 * sizeof(STREAM_TYPE) * ARRAY_SIZE, // FP16 FLOPS MVE
};

static constexpr uint32_t flop[RESULTS] = {
	0, 0, 0, // Bei Copy werden keine FLOP ausgeführt
	ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE, // Bei Scale wird ein FLOP pro Element ausgeführt
	ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE, // Bei Add wird ein FLOP pro Element ausgeführt
	2 * ARRAY_SIZE, 2 * ARRAY_SIZE, 2 * ARRAY_SIZE, // Bei Triad werden zwei FLOP pro Element ausgeführt
	2000 * ARRAY_SIZE, // Triad FLOPS
	2000 * ARRAY_SIZE, // FP16 FLOPS Scalar
	800 * ARRAY_SIZE / 4, // FP32 FLOPS Scalar
	800 * ARRAY_SIZE / 4, // FP64 FLOPS Scalar
	6400 * ARRAY_SIZE / 8 // FP16 FLOPS MVE
};

extern void checkSTREAMresults();

extern "C" {
	void stream_copy(STREAM_TYPE * __restrict a, STREAM_TYPE * __restrict c, uint32_t len);
	void stream_copy_mve(STREAM_TYPE * __restrict a, STREAM_TYPE * __restrict c, uint32_t len);
	void stream_scale(STREAM_TYPE * __restrict c, STREAM_TYPE *b, STREAM_TYPE scalar, uint32_t len);
	void stream_scale_mve(STREAM_TYPE * __restrict c, STREAM_TYPE * __restrict b, STREAM_TYPE scalar, uint32_t len);
	void stream_add(STREAM_TYPE * __restrict c, STREAM_TYPE * __restrict a, STREAM_TYPE *b, uint32_t len);
	void stream_add_mve(STREAM_TYPE * __restrict c, STREAM_TYPE * __restrict a, STREAM_TYPE *b, uint32_t len);
	void stream_triad(STREAM_TYPE * __restrict a, STREAM_TYPE *b, STREAM_TYPE * __restrict c, STREAM_TYPE scalar, uint32_t len);
	void stream_triad_mve(STREAM_TYPE * __restrict a, STREAM_TYPE *b, STREAM_TYPE * __restrict c, STREAM_TYPE scalar, uint32_t len);
	void stream_triad_mve_flops(STREAM_TYPE * __restrict a, STREAM_TYPE *b, STREAM_TYPE * __restrict c, STREAM_TYPE scalar, uint32_t len);
	void flops_scalar_fp64(double * __restrict a, double *b, double * __restrict c, double scalar, uint32_t len);
	void flops_scalar_fp32(float * __restrict a, float *b, float * __restrict c, float scalar, uint32_t len);
	void flops_scalar_fp16(float16_t * __restrict a, float16_t *b, float16_t * __restrict c, float16_t scalar, uint32_t len);
	void flops_mve_fp16(float16_t * __restrict a, float16_t *b, float16_t * __restrict c, float16_t scalar, uint32_t len);
}

int main() {
	fault_dump_enable(true);
	SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
	LPRTC::getInstance().enable();

    SEGGER_RTT_printf(0, HLINE);
    
	// Benchmark wird mehrfach ausgeführt
	uint32_t times[RESULTS][ITERATIONS];
	STREAM_TYPE scalar = 3.0f;
    for (uint32_t k = 0; k < ITERATIONS; k++) {
		/* -- COPY --*/
		times[0][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j=0; j < ARRAY_SIZE; j++)
			c[j] = a[j];
		times[0][k] = LPRTC::getInstance().getCurrentValue() - times[0][k];

		times[1][k] = LPRTC::getInstance().getCurrentValue();
		stream_copy(a, c, ARRAY_SIZE);
		times[1][k] = LPRTC::getInstance().getCurrentValue() - times[1][k];

		times[2][k] = LPRTC::getInstance().getCurrentValue();
		stream_copy_mve(a, c, ARRAY_SIZE);
		times[2][k] = LPRTC::getInstance().getCurrentValue() - times[2][k];

		/* -- SCALE --*/
		times[3][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j=0; j < ARRAY_SIZE; j++)
			b[j] = scalar*c[j];
		times[3][k] = LPRTC::getInstance().getCurrentValue() - times[3][k];

		times[4][k] = LPRTC::getInstance().getCurrentValue();
		stream_scale(c, b, scalar, ARRAY_SIZE);
		times[4][k] = LPRTC::getInstance().getCurrentValue() - times[4][k];
		
		times[5][k] = LPRTC::getInstance().getCurrentValue();
		stream_scale_mve(c, b, scalar, ARRAY_SIZE);
		times[5][k] = LPRTC::getInstance().getCurrentValue() - times[5][k];

		/* -- ADD --*/
		times[6][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j=0; j < ARRAY_SIZE; j++)
			c[j] = a[j]+b[j];
		times[6][k] = LPRTC::getInstance().getCurrentValue() - times[6][k];
		
		times[7][k] = LPRTC::getInstance().getCurrentValue();
		stream_add(c, a, b, ARRAY_SIZE);
		times[7][k] = LPRTC::getInstance().getCurrentValue() - times[7][k];

		times[8][k] = LPRTC::getInstance().getCurrentValue();
		stream_add_mve(c, a, b, ARRAY_SIZE);
		times[8][k] = LPRTC::getInstance().getCurrentValue() - times[8][k];

		/* -- TRIAD --*/
		times[9][k] = LPRTC::getInstance().getCurrentValue();
		for (uint32_t j=0; j < ARRAY_SIZE; j++)
			a[j] = b[j]+scalar*c[j];
		times[9][k] = LPRTC::getInstance().getCurrentValue() - times[9][k];

		times[10][k] = LPRTC::getInstance().getCurrentValue();
		stream_triad(a, b, c, scalar, ARRAY_SIZE);
		times[10][k] = LPRTC::getInstance().getCurrentValue() - times[10][k];

		times[11][k] = LPRTC::getInstance().getCurrentValue();
		stream_triad_mve(a, b, c, scalar, ARRAY_SIZE);
		times[11][k] = LPRTC::getInstance().getCurrentValue() - times[11][k];

		times[12][k] = LPRTC::getInstance().getCurrentValue();
		stream_triad_mve_flops(a, b, c, scalar, ARRAY_SIZE);
		times[12][k] = LPRTC::getInstance().getCurrentValue() - times[12][k];
	
		times[13][k] = LPRTC::getInstance().getCurrentValue();
		flops_scalar_fp16(fp16_array, fp16_array, fp16_array, (float16_t)scalar, ARRAY_SIZE);
		times[13][k] = LPRTC::getInstance().getCurrentValue() - times[13][k];

		times[14][k] = LPRTC::getInstance().getCurrentValue();
		flops_scalar_fp32(a, b, c, scalar, ARRAY_SIZE);
		times[14][k] = LPRTC::getInstance().getCurrentValue() - times[14][k];

		times[15][k] = LPRTC::getInstance().getCurrentValue();
		flops_scalar_fp64(fp64_array, fp64_array, fp64_array, (double)scalar, ARRAY_SIZE);
		times[15][k] = LPRTC::getInstance().getCurrentValue() - times[15][k];

		times[16][k] = LPRTC::getInstance().getCurrentValue();
		flops_mve_fp16(fp16_array, fp16_array, fp16_array, (float16_t)scalar, ARRAY_SIZE);
		times[16][k] = LPRTC::getInstance().getCurrentValue() - times[16][k];
	}

    for (uint32_t k = 1; k < ITERATIONS; k++) /* note -- skip first iteration */
	{
		for (uint32_t j = 0; j < RESULTS; j++) {
			avgtime[j] = avgtime[j] + times[j][k];
			mintime[j] = std::min(mintime[j], times[j][k]);
			maxtime[j] = std::max(maxtime[j], times[j][k]);
		}
	}
    
    SEGGER_RTT_printf(0, "Function \t    Avg MB/s\t Avg FLOPS\t Avg time\t Best MB/s\t Best FLOPS\t Min time\t Worst MB/s\t Worst FLOPS\t Max time\n");
    for (uint32_t j = 0; j < RESULTS; j++) {
		uint32_t avg = avgtime[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
		float avgSec = (float)avg / 32768.0f;
		sprintf(PRINTF_OUT_STRING, "%s\t %12.1f\t %12.1f\t %11.6f\t %12.1f\t %12.1f\t %11.6f\t %12.1f\t %12.1f\t %11.6f\n", label[j],
	    	1.0E-06 * bytes[j]/avgSec, 1.0E-06 * flop[j]/avgSec, avgSec,
		   	1.0E-06 * bytes[j]/(mintime[j]/32768.0f), 1.0E-06 * flop[j]/(mintime[j]/32768.0f), mintime[j]/32768.0f,
		   	1.0E-06 * bytes[j]/(maxtime[j]/32768.0f), 1.0E-06 * flop[j]/(maxtime[j]/32768.0f), maxtime[j]/32768.0f
		);
	   SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    }
    SEGGER_RTT_printf(0, HLINE);

	SEGGER_RTT_printf(0, "Function;AvgMB/s;AvgMFLOPS;AvgTime;BestMB/s;BestMFLOPS;MinTime;WorstMB/s;WorstMFLOPS;MaxTime\n");
    for (uint32_t j = 0; j < RESULTS; j++) {
		uint32_t avg = avgtime[j] / (ITERATIONS-1); // erste Iteration wurde ignoriert
		float avgSec = (float)avg / 32768.0f;
		sprintf(PRINTF_OUT_STRING, "%s\t %12.1f\t %12.1f\t %11.6f\t %12.1f\t %12.1f\t %11.6f\t %12.1f\t %12.1f\t %11.6f\n", label[j],
	    	1.0E-06 * bytes[j]/avgSec, 1.0E-06 * flop[j]/avgSec, avgSec,
		   	1.0E-06 * bytes[j]/(mintime[j]/32768.0f), 1.0E-06 * flop[j]/(mintime[j]/32768.0f), mintime[j]/32768.0f,
		   	1.0E-06 * bytes[j]/(maxtime[j]/32768.0f), 1.0E-06 * flop[j]/(maxtime[j]/32768.0f), maxtime[j]/32768.0f
		);
		SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    }
    SEGGER_RTT_printf(0, HLINE);

    /* --- Check Results --- */
    // checkSTREAMresults();
    SEGGER_RTT_printf(0, HLINE);
	LPRTC::getInstance().disable();
	while (1) {
		__WFE();
	}

    return 0;
}

void checkSTREAMresults ()
{
	constexpr float epsilon = sizeof(STREAM_TYPE) == 8 ? 1.e-13f : 1.0E-6f;
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	uint32_t ierr,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (uint32_t k = 0; k < ITERATIONS; k++) {
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
	}

    /* accumulate deltas between observed and expected results */
	STREAM_TYPE aSumErr = 0.0;
	STREAM_TYPE bSumErr = 0.0;
	STREAM_TYPE cSumErr = 0.0;
	for (uint32_t j = 0; j < ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}
	aAvgErr = aSumErr / static_cast<STREAM_TYPE>(ARRAY_SIZE);
	bAvgErr = bSumErr / static_cast<STREAM_TYPE>(ARRAY_SIZE);
	cAvgErr = cSumErr / static_cast<STREAM_TYPE>(ARRAY_SIZE);

	err = 0;
	if (abs(aAvgErr / aj) > epsilon) {
		err++;
		SEGGER_RTT_printf(0, "Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		SEGGER_RTT_printf(0, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (uint32_t j=0; j < ARRAY_SIZE; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
			}
		}
		SEGGER_RTT_printf(0, "     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr / bj) > epsilon) {
		err++;
		SEGGER_RTT_printf(0, "Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		SEGGER_RTT_printf(0, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		SEGGER_RTT_printf(0, "     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (uint32_t j = 0; j < ARRAY_SIZE; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
			}
		}
		SEGGER_RTT_printf(0, "     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr / cj) > epsilon) {
		err++;
		SEGGER_RTT_printf(0, "Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		SEGGER_RTT_printf(0, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		SEGGER_RTT_printf(0, "     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (uint32_t j = 0; j < ARRAY_SIZE; j++) {
			if (abs(c[j] / cj - 1.0) > epsilon) {
				ierr++;
			}
		}
		SEGGER_RTT_printf(0, "     For array c[], %d errors were found.\n",ierr);
	}
	if (err == 0) {
		SEGGER_RTT_printf(0, "Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
}
