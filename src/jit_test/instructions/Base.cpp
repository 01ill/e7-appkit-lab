#include "Base.hpp"

JIT::Instructions::Instruction16 JIT::Instructions::Base::nop() {
    return 0b1011'1111'0000'0000;
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
    instr |= (0xfe & imm11); // set immh
    return instr;
}
