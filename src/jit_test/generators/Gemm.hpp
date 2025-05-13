#ifndef JIT_GENERATORS_GEMM_HPP
#define JIT_GENERATORS_GEMM_HPP

#include "backend/Backend.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class Gemm;
    }
}

class JIT::Generators::Gemm {
    private:
        Backend backend;

    public:
        using Func = void (*) (float const *, float const *, float *);
        void (*generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc))(float const *a, float const *b, float *c);
        Func thumbAddressToFunc(uintptr_t thumbAddress) {
            __asm("dsb");
            __asm("isb");
            return reinterpret_cast<JIT::Generators::Gemm::Func>(thumbAddress);
        }
        Func bufferToFunc(Instructions::Instruction16 * buffer) {
            backend.copyToBuffer(buffer);
            return thumbAddressToFunc(backend.getBufferThumbAddress(buffer));
        }
};

#endif // JIT_GENERATORS_GEMM_HPP