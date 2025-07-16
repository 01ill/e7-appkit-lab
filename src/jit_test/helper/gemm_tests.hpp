#ifndef GEMM_TESTS_HPP
#define GEMM_TESTS_HPP

#include <cstdint>
#include "../generators/Gemm.hpp"

void initMatrices(float * a, float * b, float * c, float * cref, const uint32_t m, const uint32_t n, const uint32_t k, bool zeroC = false);
int32_t testShapeGenerateTime(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator);
int32_t testShape(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations, JIT::Generators::Gemm & generator);
int32_t testShapeReference(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations);
int32_t testShapeIntrinsics(
        float * bigA, float * bigB, float * bigC, float * bigCRef,
        uint32_t m, uint32_t n, uint32_t k, uint32_t iterations);
int32_t testShapeArm(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    uint32_t m, uint32_t n, uint32_t k, uint32_t iterations);
void testSquareShapes(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference);
void testGrowingK(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference);
void testGrowingM(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference);
void testGrowingN(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference);
void constSizeTest(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    uint32_t m, uint32_t n, uint32_t k);
void testAllSizes(
    float * bigA, float * bigB, float * bigC, float * bigCRef,
    JIT::Instructions::Instruction16 * globalBuffer,
    bool testArm, bool testJitter, bool testIntrinsics, bool testReference,
    uint32_t start, uint32_t end);
#endif // GEMM_TESTS_HPP