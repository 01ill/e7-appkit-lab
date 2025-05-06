#include "DataProcessing.hpp"
#include "instructions/Base.hpp"
#include <cstdarg>
#include <cassert>

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::ldrRegister16(Register Rn, Register Rt, Register Rm) {
    #ifdef VALIDATE_ENCODINGS
    assert(Rn <= Register::R7);
    assert(Rt <= Register::R7);
    #endif

    Instruction16 instr = 0b0110'1000'0000'0000;
    instr |= Rt;
    instr |= Rn << 3U;
    return instr;
}

JIT::Instructions::Instruction32 JIT::Instructions::DataProcessing::ldrRegister32(Register Rt,
                                                             Register Rn,
                                                             Register Rm,
                                                             uint8_t imm2) {
    Instruction32 instr = 0xf850'0000;

    #ifdef VALIDATE_ENCODINGS
    // will result in unpredictable behavior
    assert(Rm != Register::R13);
    assert(Rm != Register::R15);
    #endif

    instr |= Rm;
    instr |= Rt << 12U;
    instr |= Rn << 16U;
    instr |= (0x03 & imm2) << 4U; // Mask off Left Shift
    return instr;
}

// Low Reg Variant
JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::str(Register Rn, Register Rt) {
    Instruction16 instr = 0b0110'0000'0000'0000;
    instr |= Rt; // da Register enum ist, kann man sich mask off sparen
    instr |= Rn << 3U;
    return instr;
}

JIT::Instructions::Instruction16 JIT::Instructions::DataProcessing::movImmediate16(Register Rd, uint8_t imm8) {
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