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
        using FuncVoid = void (*) ();
        void (*generate(uint32_t operational_intensity))(uint32_t len);
        void (*generateVfma(uint32_t vfmaCount))();
        // void (*generate(uint32_t flops, uint32_t vectorCount))(float const *a, float const *b, float *c, uint32_t);
        // void (*generateNoMem(uint32_t operational_intensity))(float const *a, float const *b, float *c, uint32_t);
        // void (*generateSteps(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size);
        // void (*generateStepsNoMem(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size);
};

#endif // JIT_GENERATORS_PEAK_HPP