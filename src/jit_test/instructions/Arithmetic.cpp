#include "Arithmetic.hpp"

JIT::Instructions::Instruction32 JIT::Instructions::Arithmetic::add(Register dstReg, Register srcReg1, uint16_t imm12) {
    Instruction32 instr = 0b1111'0001'0000'0000'0000'0000'0000'0000;
    instr |= dstReg << 8U; // set Rd
    instr |= srcReg1 << 16U; // set Rn

    imm12 &= 0x0fff; // ensure 12bit
    instr |= 0x0ff & imm12; // set imm8
    instr |= (0xe & (imm12 >> 8U)) << 12U; // set imm3
    instr |= (imm12 >> 11U) << 10U; // set i
    return instr;
}