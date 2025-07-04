#ifndef JIT_GENERATORS_PEAK_HPP
#define JIT_GENERATORS_PEAK_HPP

#include "backend/Backend.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class PeakPerformance;
    }
}

class JIT::Generators::PeakPerformance {
    private:
        Backend backend;

    public:
        PeakPerformance(Instructions::Instruction16 * globalBuffer, uint32_t bufferSize) : backend(globalBuffer, bufferSize) {}
        using Func = void (*) (uint32_t);
        void (*generate(uint32_t operational_intensity))(uint32_t len);
        // void (*generate(uint32_t flops, uint32_t vectorCount))(float const *a, float const *b, float *c, uint32_t);
        // void (*generateNoMem(uint32_t operational_intensity))(float const *a, float const *b, float *c, uint32_t);
        // void (*generateSteps(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size);
        // void (*generateStepsNoMem(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size);
        // Func thumbAddressToFunc(uintptr_t thumbAddress) {
        //     __asm("dsb");
        //     __asm("isb");
        //     return reinterpret_cast<JIT::Generators::PeakPerformance::Func>(thumbAddress);
        // }
        // Func bufferToFunc(Instructions::Instruction16 * buffer) {
        //     backend.copyToBuffer(buffer);
        //     return thumbAddressToFunc(backend.getBufferThumbAddress(buffer));
        // }
};

#endif // JIT_GENERATORS_PEAK_HPP