#include "DataProcessing.hpp"
#include "instructions/Base.hpp"
#include <cstdarg>

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::ldr(Register Rn, Register Rt) {
    Instruction16 instr = 0b0110'1000'0000'0000;
    instr |= Rt;
    instr |= Rn << 3U;
    return instr;
}
/*
JIT::Instructions::Instruction32 JIT::Instructions::DataProcessing::ldr(Register Rn, Register Rt) {
    Instruction32 instr = 0xf8d0'0000;
    instr |= Rn << 16U;
    instr |= Rt << 12U;
    return instr;
}
*/

// Low Reg Variant
JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::str(Register Rn, Register Rt) {
    Instruction16 instr = 0b0110'0000'0000'0000;
    instr |= Rt; // da Register enum ist, kann man sich mask off sparen
    instr |= Rn << 3U;
    return instr;
}

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::mov(Register Rd, uint8_t imm8) {
    Instruction16 instr = 0b0010'0000'0000'0000;
    instr |= imm8;
    instr |= Rd << 8U;
    return instr;
}

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::push(Register const Rd) {
    Instruction16 instr = 0b1011'0100'0000'0000;

    if (Rd == Register::LR) {
        instr |= 1U << 8U; // LR -> Bit 8 auf 1
    } else {
        instr |= 1U << Rd;
    }
    return instr;
}

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::pop(Register const Rd) {
    Instruction16 instr = 0b1011'1100'0000'0000;

    if (Rd == Register::PC) {
        instr |= 1U << 8U; // PC -> Bit 8 auf 1
    } else {
        // Man hat für jedes Low Register ein Bit von Bit 0-7. Wenn ein Bit davon auf 1 gesetzt wird, wird das Register gepopt
        instr |= 1U << Rd;
    }

    return instr;
}