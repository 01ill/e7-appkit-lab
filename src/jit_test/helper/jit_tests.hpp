#ifndef JIT_TESTS_HPP
#define JIT_TESTS_HPP
#include <cstdint>
#include "../backend/Backend.hpp"

void testPeakPerformance(JIT::Instructions::Instruction16 * globalBuffer, JIT::Instructions::Instruction16 * globalBufferSram0, JIT::Instructions::Instruction16 * globalBufferDtcm, uint32_t arrayMaxSize);
void testThroughput(JIT::Instructions::Instruction16 * globalBuffer, JIT::Instructions::Instruction16 * globalBufferSram0, JIT::Instructions::Instruction16 * globalBufferDtcm, uint32_t arrayMaxSize, float * bigA);

#endif // JIT_TESTS_HPP
