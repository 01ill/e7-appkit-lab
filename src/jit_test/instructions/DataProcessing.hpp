#ifndef JIT_INSTRUCTIONS_DATA_PROCESSING_HPP
#define JIT_INSTRUCTIONS_DATA_PROCESSING_HPP

#include "Base.hpp"
#include <cstdint>

namespace JIT {
    namespace Instructions {
        class DataProcessing;
    }
}

class JIT::Instructions::DataProcessing {
    public:
        static Instruction16 ldr(Register Rn, Register Rt);
        // static Instruction32 ldr(Register Rn, Register Rt);
        static Instruction16 str(Register Rn, Register Rt);
        static Instruction16 mov(Register Rd, uint8_t imm8);
        static Instruction16 push(Register Rd);
        static Instruction16 pop(Register Rd);
};

#endif // JIT_INSTRUCTIONS_DATA_PROCESSING_HPP