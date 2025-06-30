#ifndef JIT_GENERATORS_THROUGHPUT_HPP
#define JIT_GENERATORS_THROUGHPUT_HPP

#include "backend/Backend.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class Throughput;
    }
}

class JIT::Generators::Throughput {
    private:
        Backend backend;

    public:
        using Func = void (*) (float const *, uint32_t);
        void (*generate())(float const *a, uint32_t len);
        Func thumbAddressToFunc(uintptr_t thumbAddress) {
            __asm("dsb");
            __asm("isb");
            return reinterpret_cast<JIT::Generators::Throughput::Func>(thumbAddress);
        }
        Func bufferToFunc(Instructions::Instruction16 * buffer) {
            backend.copyToBuffer(buffer);
            return thumbAddressToFunc(backend.getBufferThumbAddress(buffer));
        }


};

#endif // JIT_GENERATORS_THROUGHPUT_HPP