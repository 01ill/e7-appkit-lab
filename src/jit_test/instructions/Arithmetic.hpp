#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include "Base.hpp"

namespace JIT {
    namespace Instructions {
        class Arithmetic;
    }
}

class JIT::Instructions::Arithmetic {
    public:
        static Instruction32 add(Register dstReg, Register srcReg1, Register srcReg2);
        static Instruction32 add(Register dstReg, Register srcReg1, uint16_t imm12);
};

#endif // ARITHMETIC_HPP