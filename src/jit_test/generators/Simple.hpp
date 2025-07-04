#ifndef SIMPLE_HPP
#define SIMPLE_HPP

#include "../backend/Backend.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class Simple;
    }
}

class JIT::Generators::Simple {
    private:
        Backend backend;

    public:
        Simple(Instructions::Instruction16 * globalBuffer, uint32_t bufferSize) : backend(globalBuffer, bufferSize) {}
        using Func = uint32_t (*) ();
        uint32_t (*generate())();
};

#endif