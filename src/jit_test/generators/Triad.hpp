#ifndef JIT_GENERATORS_TRIAD_HPP
#define JIT_GENERATORS_TRIAD_HPP

#include "backend/Backend.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class Triad;
    }
}

class JIT::Generators::Triad {
    private:
        Backend backend;

    public:
        using Func = void (*) (float const *, float const *, float *, float const);
        void (*generate(uint32_t count))(float const *a, float const *b, float *c, float const);
};

#endif // JIT_GENERATORS_TRIAD_HPP