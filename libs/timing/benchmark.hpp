#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstdint>

#ifdef USE_CMSIS_DSP
#include "arm_math.h"
#endif

void setupTests();
void stopTests();

template <typename Func, typename Result, typename A, typename B, typename C>
uint32_t benchmark(const Func f, const uint32_t iterations, Result result, A a, B b, C c, uint32_t len);

// additional benchmark for CMSIS-DSP functions
// only enabled if the USE_CMSIS_DSP flag is set via the cproject file
// the ARM::CMSIS:DSP component needs to be included in the cproject file
// as the arm_math.h file is only included in the DSP library
#ifdef USE_CMSIS_DSP
template <typename Func>
uint32_t benchmarkArm(const Func f, const uint32_t iterations, arm_status* status, arm_matrix_instance_f32 * a, arm_matrix_instance_f32 * b, arm_matrix_instance_f32 * c);
#endif

template <typename T>
int32_t compare(T resultA, T resultB, uint32_t size);


#endif // BENCHMARK_H
