#include "gemm_tests.hpp"
#include <chrono>
#include <cstdint>
#include "profiling.hpp"
#include "timing.hpp"
#include "gemm_kernel.hpp"
#include "benchmark.hpp"

#ifdef M55_HE
constexpr float peak = 0.64;
constexpr uint32_t arrSize = 100;
#endif

#ifdef M55_HP
constexpr float peak = 1.6;
constexpr uint32_t arrSize = 240;
#endif
static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));


void initMatrices(float * a, float * b, float * c, float * cref, const uint32_t m, const uint32_t n, const uint32_t k, bool zeroC, bool useFloat) {
    for (uint32_t i = 0; i < m*k; i++) a[i] = i + (useFloat ? (i / (i+1.0f)) : 0);
    for (uint32_t i = 0; i < k*n; i++) b[i] = i + 1 + (useFloat ? (i / (i - 100.0f)) : 0);
    for (uint32_t i = 0; i < m*n; i++) c[i] = zeroC ? 0 : i + 2 + (useFloat ? 0.124974f : 0);
	for (uint32_t i = 0; i < m*n; i++) cref[i] = zeroC ? 0 : i + 2 + (useFloat ? 0.124974f : 0);
}

int32_t testShapeGenerateTime(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator) {
    JIT::Generators::Gemm::Func gemmFunc;
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);

    auto start = RTC_Clock::now();
    for (uint32_t it = 0; it < iterations; it++) {
        gemmFunc = generator.generate(m, k, n, m, k, m);
    }
    auto end = RTC_Clock::now();
    gemmFunc(bigA, bigB, bigC);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return time; // return negative value if test not succesful
}

int32_t testShape(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator) {
    auto gemmFunc = generator.generate(m, k, n, m, k, m);
    // auto gemmFunc = generator.generate(mBlocking, kBlocking, n, m, k, m);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    // setupProfilingMVEInstructions();
    // startCounting();
    gemmFunc(bigA, bigB, bigC);
    // stopCounting();
    // printCounterMVEInstructions();

    gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    int32_t compareResult = compare(bigC, bigCRef, m*n);
    if (iterations == 0) {
        if (compareResult != -1) SEGGER_RTT_printf(0, "Fail at %d;", compareResult);
        return compareResult != -1 ? -compareResult : 1; // negative value == fail
    } else {
        initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
        auto start = RTC_Clock::now();
        for (uint32_t it = 0; it < iterations; it++) {
            // jitBlocked_mk(bigA, bigB, bigC, m, n, k, gemmFunc);
            gemmFunc(bigA, bigB, bigC);
        }
        auto end = RTC_Clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (time == 0) time = 1;
        if (compareResult != -1) SEGGER_RTT_printf(0, "Fail at %d;", compareResult);
        return compareResult != -1 ? -time : time; // return negative value if test not succesful
    }
}

int32_t testShapeReference(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    }
    auto end = RTC_Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int32_t testShapeIntrinsics(
        float * bigA, float * bigB, float * bigC, float * bigCRef,
        uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    gemm_intrinsics_8x3(bigA, bigB, bigC, n, k, m, m, k, m);
    gemm_reference_column_major(bigA, bigB, bigCRef, n, k, m, m, k, m);
    int32_t compareResult = compare(bigC, bigCRef, m*n);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        gemm_intrinsics_8x3(bigA, bigB, bigCRef, n, k, m, m, k, m);
    }
    auto end = RTC_Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return compareResult != -1 ? -time : time; // return negative value if test not succesful
}


int32_t testShapeArm(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations) {
    arm_matrix_instance_f32 armA;
    arm_matrix_instance_f32 armB;
    arm_matrix_instance_f32 armC;
    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k, true);
    arm_mat_init_f32(&armA, m, k, bigA);
    arm_mat_init_f32(&armB, k, n, bigB);
    arm_mat_init_f32(&armC, m, n, bigC);
    gemm_reference_row_major(bigA, bigB, bigCRef, n, k, m, k, n, n);
    arm_mat_mult_f32(&armA, &armB, &armC);
    int32_t compareResult = compare(bigC, bigCRef, m*n);

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k, true);
    auto start = RTC_Clock::now();
    for (uint32_t j = 0; j < iterations; j++) {
        arm_mat_mult_f32(&armA, &armB, &armC);
    }
    auto end = RTC_Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return compareResult != -1 ? -time : time; // return negative value if test not succesful
}

void testSquareShapes(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference) {
    int32_t time;
    double gflops;
    uint32_t m, n, k;
    JIT::Generators::Gemm gemmGen(globalBuffer, 3072);
    SEGGER_RTT_printf(0, "--- START TEST SQUARE SHAPES ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 1; i <= 240; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        m = i;
        n = i;
        k = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        iterations = iterations > 10000000 ? 10000000 : iterations;
        // iterations = 30000;
        if (testArm) {
            time = testShapeArm(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
        
        if (testIntrinsics && m % 8 == 0 && n % 3 == 0) {
            time = testShapeIntrinsics(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;Intrinsics;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST SQUARE SHAPES ---\n\n");
}

void testGrowingK(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference) {
    int32_t time;
    double gflops;
    uint32_t m = 24, n = 24, k;
    JIT::Generators::Gemm gemmGen(globalBuffer, 3072);
    SEGGER_RTT_printf(0, "--- START TEST GROWING K ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 1; i <= arrSize; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        k = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        iterations = iterations > 10000000 ? 10000000 : iterations;
        if (testArm) {
            time = testShapeArm(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics) {
            time = testShapeIntrinsics(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowK;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING K ---\n\n");
}

void testGrowingM(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference) {

    int32_t time;
    double gflops;
    uint32_t m = 1, n = 24, k = 24;
    JIT::Generators::Gemm gemmGen(globalBuffer, 3072);
    SEGGER_RTT_printf(0, "--- START TEST GROWING M ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 1; i <= arrSize; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        m = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        iterations = iterations > 10000000 ? 10000000 : iterations;
        if (testArm) {
            time = testShapeArm(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics && m % 8 == 0) {
            time = testShapeIntrinsics(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowM;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING M ---\n\n");
}

void testGrowingN(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference) {

    int32_t time;
    double gflops;
    uint32_t m = 24, n = 1, k = 24;
    JIT::Generators::Gemm gemmGen(globalBuffer, 3072);
    SEGGER_RTT_printf(0, "--- START TEST GROWING N ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t i = 1; i <= arrSize; i++) {
        // run approximately one second assuming peak performance
        // 1.6gflops => 2n^3 flop per iteration
        // 2n^3 * x = 1.6 * 10^9
        n = i;
        uint32_t flops = 2 * m * k * n;
        uint32_t iterations = (peak * pow(10, 9)) / flops;
        if (testArm) {
            time = testShapeArm(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testJitter) {
            time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testIntrinsics && n % 3 == 0) {
            time = testShapeIntrinsics(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            bool correctResult = gflops > 0;
            if (!correctResult) gflops = -gflops;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;Intrinsics;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }

        if (testReference) {
            iterations /= 10;
            time = testShapeReference(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
            gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
            sprintf(PRINTF_OUT_STRING, "GrowN;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
            SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST GROWING N ---\n\n");
}

void constSizeTest(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    uint32_t m, uint32_t n, uint32_t k, bool validate) {

    initMatrices(bigA, bigB, bigC, bigCRef, m, n, k);
    JIT::Generators::Gemm gemmGen(globalBuffer, 10000);
    uint32_t repeats = 5;
    uint32_t flops = 2 * m * k * n;
    uint32_t iterations = (peak * pow(10, 9)) / flops;
    if (validate) iterations = 0;
    int32_t time;
    double gflops;
    for (uint32_t i = 0; i < repeats; i++) {
        time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
        gflops = (static_cast<float>(flops) / (time * 1000000.0f)) * iterations;
        sprintf(PRINTF_OUT_STRING, "UnrollN;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, gflops > 0);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
    }
}

void testAllSizes(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference,
    uint32_t start, uint32_t end, uint32_t resume, bool validate) {
    int32_t time;
    double gflops;
    JIT::Generators::Gemm gemmGen(globalBuffer, 3072);
    SEGGER_RTT_printf(0, "--- START TEST SQUARE SHAPES ---\n");
    SEGGER_RTT_printf(0, "Test;M;K;N;Type;GFLOPS;Time;Iterations;Correct\n");
    for (uint32_t m = start; m <= end; m++) {
        for (uint32_t n = start; n <= end; n++) {
            for (uint32_t k = start; k <= end; k++) {
                if (m < resume && n < resume && k < resume) continue;
                // run approximately one second assuming peak performance
                // 1.6gflops => 2n^3 flop per iteration
                // 2n^3 * x = 1.6 * 10^9
                uint32_t flops = 2 * m * k * n;
                uint32_t iterations = ((peak * pow(10, 9)) / flops) * 0.2;
                iterations = iterations > 10000000 ? 10000000 : iterations;
                if (validate) iterations = 0;
                if (testArm) {
                    time = testShapeArm(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
                    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
                    sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ARM-CMSIS-DSP;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
                    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
                }

                if (testJitter) {
                    time = testShape(bigA, bigB, bigC, bigCRef, m, n, k, iterations, gemmGen);
                    gflops = static_cast<float>(flops) / (time * 1000000.0f) * iterations;
                    bool correctResult = gflops > 0;
                    if (!correctResult) gflops = -gflops;
                    sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;TillJIT;%f;%d;%d;%d\r\n", m, k, n, gflops, time, iterations, correctResult);
                    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
                }
                
                if (testIntrinsics && m % 8 == 0 && n % 3 == 0) {
                    time = testShapeIntrinsics(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
                    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
                    bool correctResult = gflops > 0;
                    if (!correctResult) gflops = -gflops;
                    sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;Intrinsics;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
                    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
                }

                if (testReference) {
                    iterations /= 10;
                    time = testShapeReference(bigA, bigB, bigC, bigCRef, m, n, k, iterations);
                    gflops = static_cast<float>(flops) / (time/1000.0f * pow(10, 9)) * iterations;
                    sprintf(PRINTF_OUT_STRING, "Square;%d;%d;%d;ReferenceCM;%f;%d;%d;1\r\n", m, k, n, gflops, time, iterations);
                    SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);
                }
            }
        }
    }
    SEGGER_RTT_printf(0, "--- END TEST ALL SIZES ---\n\n");

}