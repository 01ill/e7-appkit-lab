#include "Base.hpp"
#include <cstdint>

using namespace JIT::Instructions;

JIT::Instructions::Instruction16 JIT::Instructions::Base::nop16() {
    return 0xbf00;
}

Instruction32 Base::nop32() {
    return 0xf3af'8000;
}

JIT::Instructions::Instruction16 JIT::Instructions::Base::bx(Register Rm) {
    Instruction16 instr = 0b0100011100000000;
    instr |= Rm << 3U;
    return instr;
}

JIT::Instructions::Instruction32 JIT::Instructions::Base::dlstp(Register Rn, Size size) {
    Instruction32 instr = 0xF000'E001;
    instr |= Rn << 16U;
    instr |= size << 20U;

    return instr;
}

// jumpAddr = PC - imm32
// imm32 = ZeroExtend(immh:imml:'0', 32)
JIT::Instructions::Instruction32  JIT::Instructions::Base::letp(int16_t imm11) {
    Instruction32 instr = 0xF01F'C001;

    imm11 = -imm11; // wird von PC abgezogen, d.h. wenn imm11 negativ ist, dann muss es um zurückzuspringen positiv gesetzt werden
    imm11 >>= 1U; // right shift, weil im Decode ein "Left Shift" gemacht wird
    instr |= (0x01 & imm11) << 11U; // set imml
    instr |= (0x7fe & imm11); // set immh
    return instr;
}

Instruction16 Base::cmpImmediate16(Register Rn, uint8_t imm8) {
    Instruction16 instr = 0x2800;
    instr |= imm8;
    instr |= Rn << 8;
    return instr;
}

Instruction16 Base::cmpRegister16(Register Rn, Register Rm) {
    if (Rn > 7 || Rm > 7) { // At least one high register
        Instruction16 instr = 0x4500;
        instr |=  0x7 & Rn; // set Rn
        instr |= (Rn >> 3) << 7; // set N
        instr |= Rm << 3;
        return instr;
    } else { // Both low registers
        Instruction16 instr = 0x4280;
        instr |= Rn;
        instr |= Rm << 3;
        return instr;
    }
}

// amount = imm3:imm2
Instruction32 Base::cmpRegister32(Register Rn, Register Rm, Shift shift, uint8_t amount) {
    Instruction32 instr = 0xebb0'0f00;
    instr |= Rm;
    instr |= shift << 4;
    instr |= (0x3 & amount) << 6; // set imm2
    instr |= (amount >> 2) << 12; // set imm3
    instr |= Rn << 16;
    return instr;
}

Instruction16 Base::bCond16(Condition cond, int16_t imm8) {
    Instruction16 instr = 0xd000;
    if (imm8 == 2) {
        instr |= 0xff;
    } else {
        imm8 -= 4;
        instr |= (0x1ff & imm8) >> 1; // imm32 = imm8:0    
    }
    instr |= cond << 8;
    return instr;
}

Instruction16 Base::bCond16(int16_t imm11) {
    Instruction16 instr = 0xe000;
    instr |= (0x7ff & imm11) >> 1; // set imm11
    return instr;
}