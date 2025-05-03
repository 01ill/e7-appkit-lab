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

        /**
         * @brief Calculates address from base register value (Rn) and an offset value (Rm),
         * loads word from memory and writes it to target register (Rt).
         * Offset can be shifted left by 0-3 bits.
         * @param Rt Target Register
         * @param Rn Base Address Register
         * @param Rm Offset Register
         * @param imm2 Optional Left Shift (0-3) for the offset. Default is 0.
         * @return Instruction32 Encoded Instruction
         * @see C2.4.80, Encoding T2, p. 673
        */
        static Instruction32 ldrRegister(Register Rt, Register Rn, Register Rm, uint8_t imm2 = 0);

        static Instruction16 str(Register Rn, Register Rt);
        static Instruction16 mov(Register Rd, uint8_t imm8);
        static Instruction16 push(Register Rd);
        static Instruction16 pop(Register Rd);
};

#endif // JIT_INSTRUCTIONS_DATA_PROCESSING_HPP