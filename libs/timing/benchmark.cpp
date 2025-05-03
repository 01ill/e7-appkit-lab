// #define USE_GPIO

#include <cstdint>
#ifdef USE_GPIO
extern "C" {
    #include "board.h"
    #include "Driver_GPIO.h"
}
#endif

#include <chrono>
#include "timing.hpp"
#include "benchmark.hpp"

#ifdef USE_CMSIS_DSP
#include "arm_math.h"
#endif
#ifdef USE_GPIO
#define _GET_DRIVER_REF(ref, peri, chan) \
    extern ARM_DRIVER_##peri Driver_##peri##chan; \
    static ARM_DRIVER_##peri * ref = &Driver_##peri##chan;
#define GET_DRIVER_REF(ref, peri, chan) _GET_DRIVER_REF(ref, peri, chan)

GET_DRIVER_REF(gpio_b, GPIO, BOARD_LEDRGB0_B_GPIO_PORT);
GET_DRIVER_REF(gpio_r, GPIO, BOARD_LEDRGB0_R_GPIO_PORT);
#endif

void setupTests() {
    RTC_Initialize();
    #ifdef USE_GPIO
    gpio_b->Initialize(BOARD_LEDRGB0_B_PIN_NO, NULL);
    gpio_b->PowerControl(BOARD_LEDRGB0_B_PIN_NO, ARM_POWER_FULL);
    gpio_b->SetDirection(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);

    gpio_r->Initialize(BOARD_LEDRGB0_R_PIN_NO, NULL);
    gpio_r->PowerControl(BOARD_LEDRGB0_R_PIN_NO, ARM_POWER_FULL);
    gpio_r->SetDirection(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
    #endif
}

void stopTests() {
    RTC_Uninitialize();
    #ifdef USE_GPIO
    gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
    gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);

    gpio_b->Uninitialize(BOARD_LEDRGB0_B_PIN_NO);
    gpio_r->Uninitialize(BOARD_LEDRGB0_R_PIN_NO);
    #endif
}

/**
 * @return Gibt Zeit in ms zurück

 * @see https://learn.microsoft.com/en-us/cpp/cpp/ellipses-and-variadic-templates?view=msvc-170
*/
template <typename Func, typename Result, typename A, typename B, typename C>
uint32_t benchmark(const Func f, const uint32_t iterations, Result result, const A a, const B b, C c, const uint32_t len) {
    #ifdef USE_GPIO
    gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_HIGH);
    #endif
    RTC_Clock::time_point start = RTC_Clock::now();
    for (uint32_t i = iterations; i > 0; --i) {
        // https://www.heise.de/blog/C-Core-Guidelines-Regeln-fuer-Variadic-Templates-4259632.html
        *result = f(a, b, c, len);
        // Ansonsten optimiert der Compiler die Assembly-Methoden weg (bei -O3)
        asm volatile("":::"memory");
    }
    RTC_Clock::time_point end = RTC_Clock::now();
    #ifdef USE_GPIO
    gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
    #endif
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

#ifdef USE_CMSIS_DSP
template<typename Func>
uint32_t benchmarkArm(const Func f, const uint32_t iterations, arm_status* result, arm_matrix_instance_f32 * a, arm_matrix_instance_f32 * b, arm_matrix_instance_f32 * c) {
    RTC_Clock::time_point start = RTC_Clock::now();
    for (uint32_t i = iterations; i > 0; --i) {
        // https://www.heise.de/blog/C-Core-Guidelines-Regeln-fuer-Variadic-Templates-4259632.html
        *result = f(a, b, c);
        // Ansonsten optimiert der Compiler die Assembly-Methoden weg (bei -O3)
        asm volatile("":::"memory");
    }
    RTC_Clock::time_point end = RTC_Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}
template uint32_t benchmarkArm<arm_status (*)(arm_matrix_instance_f32 const*, arm_matrix_instance_f32 const*, arm_matrix_instance_f32*)>(
    arm_status (*)(arm_matrix_instance_f32 const*, arm_matrix_instance_f32 const*, arm_matrix_instance_f32*), uint32_t, arm_status*, arm_matrix_instance_f32*, arm_matrix_instance_f32*, arm_matrix_instance_f32*);
#endif

template <typename T>
int32_t compare(T resultA, T resultB, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        if (abs(resultA[i] - resultB[i]) > 0.00001) {
            return i;
        }
    }
    return -1;
}

//template int32_t benchmark<int32_t (*)(const float*, const float*, float, int), int32_t, )

//template bool compare<int>(int*, int*, int32_t);
// Func: uint32_t DOTP(float const*, float const*, float, uint32_t)
// Result: int32_t
// Arguments: float* const, float* const, float, uint32_t
// Vector*Vector -> Scalar/Vector
//                                     Func                                            Result   A       B       C 
template uint32_t benchmark<float (*)(float const*, float const*, float*, uint32_t), float*, float*, float*, float*>(
//                                         f                                         iterations result    a      b       c      len
                            float (*)(float const*, float const*, float*, uint32_t), uint32_t, float*, float*, float*, float*, uint32_t);
//template uint32_t benchmark<void (*)(float const*, float const*, float*, uint32_t), float*, float*, float*, float*>(
//                            void (*)(float const*, float const*, float*, uint32_t), uint32_t, float*, float*, float*, float*, uint32_t);


template int32_t compare<float*>(float *a, float *b, uint32_t size);