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
        enum RegisterImmediateStrategy : uint8_t {
            /*
            Default Strategy. can be used when
            - C can be done with immediates (M small enough <= 63) and
            - CMP can be done with immediates (N (<= 255) and M (<= 255) small enough)
            */
            USE_K_LEN_REGISTER, // Default Strategy: this means use C_ROW1_Base and C_ROW2_Base if C cant use immediates
            /*
            Can be used when
            - C can be done (at least partly) with immediates (M small enough <= 127)

            Shoud be used when
            - CMP can't be done with immediates because N is too large
            */
            USE_K_N_LEN_REGISTER,
            /*
            Can be used when
            - C can be done (at least partly) with immediates (M small enough <= 127)

            Shoud be used when
            - A cant be loaded because M is too big (>= 127)
            */
            USE_K_M_LEN_REGISTER, // this means use C_ROW2_Base if C cant use ummediates (C[ldc + 16] has to be done with immediates)
            USE_K_N_M_LEN_REGISTER
        };
        void generateMicroKernel(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, RegisterImmediateStrategy strategy, bool rewind = false);
        void addImmediate(JIT::Instructions::Register reg, uint32_t immediate, JIT::Instructions::Register tempReg = Instructions::PC);

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