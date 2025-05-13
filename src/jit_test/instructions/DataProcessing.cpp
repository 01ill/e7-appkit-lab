#include "DataProcessing.hpp"
#include "instructions/Base.hpp"
#include <cstdarg>
#include <cassert>
#include <cstdint>

using namespace JIT::Instructions;

Instruction16 DataProcessing::ldrImmediate16(Register Rt, Register Rn, uint8_t imm5) {
    if (!Base::assertLowRegister(Rt, Rn)) {
        Base::printValidationError("ldrImmediate16: only low registers allowed - returning nop");
        return Base::nop16();
    }
    if ((imm5 & 0x80) != 0 || (imm5 & 0x03) != 0) {
        Base::printValidationError("ldrImmediate16: immediate must be <= 124 and multiple of 4 - inserting nop");
        return Base::nop16();
    }
    Instruction16 instr = 0x6800;
    instr |= Rt;
    instr |= Rn << 3;
    instr |= (0x1f & (imm5 >> 2)) << 6;
    return instr;
}
/*
Instruction32 DataProcessing::ldrImmediate32(Register Rt, Register Rn, uint16_t imm12) {
    if ((imm12 & 0xf000) != 0) {
        Base::printValidationError("ldrImmediate32: immediate must contain <= 12bits - inserting nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xf8d0'0000;
    instr |= 0xfff & imm12;
    instr |= Rt << 12;
    instr |= Rn << 16;
    return instr;
}*/

Instruction32 DataProcessing::ldrImmediate32(Register Rt, Register Rn, int16_t imm, bool preIndexed, bool writeBack) {
    // Use Encoding T3
    if (preIndexed && imm >= 0 && !writeBack) {
        if ((imm & 0xf000) != 0) {
            Base::printValidationError("ldrImmediate32: immediate must contain <= 12bits - inserting nop");
            return Base::nop32();
        }
        Instruction32 instr = 0xf8d0'0000;
        instr |= 0xfff & imm;
        instr |= Rt << 12;
        instr |= Rn << 16;
        return instr;
    } else { // Use Encoding T4
        if (imm < -255 || imm > 255) {
            Base::printValidationError("ldrImmediate32: immediate must fit in 8bits - inserting nop");
            return Base::nop32();
        }
        if (!preIndexed && !writeBack) {
            Base::printValidationError("ldrImmediate32: post index must write back - setting write back");
            writeBack = true;
        }
        Instruction32 instr = 0xf850'0a00;
        if (imm < 0) { // negative immediate
            imm = -imm;
            instr &= 0xffff'fdff; // clear U = add bit
        }
        instr |= 0xff & imm;
        instr |= writeBack << 8;
        instr |= preIndexed << 10;
        instr |= Rt << 12;
        instr |= Rn << 16;
        return instr;
    }
}

Instruction16 DataProcessing::ldrRegister16(Register Rt, Register Rn, Register Rm) {
    if (!Base::assertLowRegister(Rt, Rn, Rm)) {
        Base::printValidationError("ldrRegister16: only low registers allowed - returning nop");
        return Base::nop16();
    }

    Instruction16 instr = 0x5800;
    instr |= Rt;
    instr |= Rn << 3U;
    instr |= Rm << 6U;
    return instr;
}

Instruction32 DataProcessing::ldrRegister32(Register Rt, Register Rn, Register Rm, uint8_t imm2) {
    if ((0xfc & imm2) != 0) {
        Base::printValidationError("ldrRegister32: shift must be between 0-3 - returning nop");
        return Base::nop32();
    }
    if (Rm == SP || Rm == PC) {
        Base::printValidationError("ldrRegister32: SP and PC not allowed as Rm - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xf850'0000;
    instr |= Rm;
    instr |= Rt << 12U;
    instr |= Rn << 16U;
    instr |= (0x03 & imm2) << 4U; // Mask off Left Shift
    return instr;
}

// Low Reg Variant
Instruction16 DataProcessing::str(Register Rn, Register Rt) {
    Instruction16 instr = 0b0110'0000'0000'0000;
    instr |= Rt; // da Register enum ist, kann man sich mask off sparen
    instr |= Rn << 3U;
    return instr;
}

Instruction16 DataProcessing::movImmediate16(Register Rd, uint8_t imm8) {
    if (!Base::assertLowRegister(Rd)) {
        Base::printValidationError("movImmediate16: only low registers allowed - returning nop");
        return Base::nop16();
    }

    Instruction16 instr = 0x2000;
    instr |= imm8;
    instr |= Rd << 8U;
    return instr;
}

// imm16 = imm4:i:imm3:imm8
Instruction32 DataProcessing::movImmediate32(Register Rd, uint16_t imm16) {
    Instruction32 instr = 0xf240'0000;
    instr |= 0xff & imm16; // set imm8
    instr |= (0x7 & (imm16>>8)) << 12; // set imm3
    instr |= (0x1 & (imm16>>11)) << 26; // set i
    instr |= (0xf & (imm16>>12)) << 16; // set imm4
    instr |= Rd << 8;
    return instr;
}

Instruction16 DataProcessing::movRegister16(Register Rd, Register Rm, Shift shift, uint8_t amount) {
    if (shift == LSL && amount == 0) {
        Instruction16 instr = 0x4600;
        instr |= 0x7 & Rd;
        instr |= (0x1 & (Rd>>3)) << 7; // d = D:Rd
        instr |= Rm << 3;
        return instr;
    }
    if (shift == ROR) {
        Base::printValidationError("movRegister16: ROR not allowed - returning nop");
        return Base::nop16();
    }
    if (!Base::assertLowRegister(Rd, Rm)) {
        Base::printValidationError("movRegister16: only low registers allowed if shift is used - returning nop");
        return Base::nop16();
    }
    if (amount > 0b11111) {
        Base::printValidationError("movRegister16: shift amount too large (max. 31) - returning nop");
        return Base::nop16();
    }

    Instruction16 instr = 0x0000;
    instr |= Rd;
    instr |= Rm << 3;
    instr |= (0x1f & amount) << 6;
    instr |= shift << 11;
    return instr;
}

// amount = imm3:imm2
Instruction32 DataProcessing::movRegister32(Register Rd, Register Rm, Shift shift, uint8_t amount, bool setFlags) {
    if (shift == ROR && amount > 0) {
        Base::printValidationError("movRegister32: RRX/ROR not allowed with extra shift - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xea4f'0000;
    instr |= Rm;
    instr |= Rd << 8;
    instr |= shift << 4;
    instr |= (0x3 & amount) << 6; // imm2
    instr |= (0x7 & (amount >> 2)) << 12;
    instr |= setFlags << 20; // set S
    return instr;
}

Instruction32 DataProcessing::vpush(DoubleRegister startRegister, uint8_t registerCount) {
    if (startRegister + registerCount > 16) {
        Base::printValidationError("vpush: Only 16 Registers can be pushed - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xed2d'0b00;
    instr |= (0xf & startRegister) << 12;
    instr |= (startRegister >> 4) << 23; // d = D:Vd
    instr |= (0x7f & registerCount) << 1; // imm8 is set to double the number of registers
    return instr;
}

Instruction32 DataProcessing::vpush(VectorRegister startRegister, uint8_t registerCount) {
    if (startRegister + registerCount > 8) {
        Base::printValidationError("vpush: Only 8 Q-registers can be pushed - returning nop");
        return Base::nop32();
    }
    return vpush(static_cast<DoubleRegister>(startRegister * 2), registerCount*2);
}

Instruction32 DataProcessing::vpop(DoubleRegister startRegister, uint8_t registerCount) {
    if (startRegister + registerCount > 16) {
        Base::printValidationError("vpop: Only 16 Registers can be pushed - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xecbd'0b00;
    instr |= (0xf & startRegister) << 12;
    instr |= (startRegister >> 4) << 23; // d = D:Vd
    instr |= (0x7f & registerCount) << 1; // imm8 is set to double the number of registers
    return instr;
}

Instruction32 DataProcessing::vpop(VectorRegister startRegister, uint8_t registerCount) {
    return vpop(static_cast<DoubleRegister>(startRegister * 2), registerCount * 2);
}