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
        using Func = uint32_t (*) ();
        uint32_t (*generate())();
};

#endif