#include "Base.hpp"
#include <cstdint>
#include <cstdio>

using namespace JIT::Instructions;

/*bool Base::assertLowRegister(Register reg) {
    return reg <= 7;
}*/

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

Instruction32 Base::dls(Register Rn) {
    Instruction32 instr = 0xf040'e001;
    instr |= Rn << 16;
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

// jumpAddr = PC - imm32
// imm32 = ZeroExtend(immh:imml:'0', 32)
Instruction32 Base::le(int16_t imm11) {
    Instruction32 instr = 0xf00f'c001;

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

Instruction32 Base::cmpImmediate32(Register Rn, uint32_t constant) {
    Instruction32 instr = 0xf1b0'0f00;

    if (!canEncodeImmediateConstant(constant)) {
        Base::printValidationError("cmpImmediate32: constant can't be encoded - returning nop");
        return Base::nop32();
    }
    instr |= Rn << 16;

    return Base::encodeImmediateConstant(instr, constant);
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
    if (cond == AL) {
        return b16(imm8);
    }
    if (imm8 > 254 || imm8 < -256) {
        Base::printValidationError("bCond16: imm8 must be in range [-256, 254] - returning nop");
        return Base::nop16();
    }
    Instruction16 instr = 0xd000;
    /*if (imm8 == 2) {
        instr |= 0xff;
    } else {
        imm8 -= 4;
        instr |= (0x1ff & imm8) >> 1; // imm32 = imm8:0    
    }*/
    instr |= (0x1ff & imm8) >> 1; // imm32 = imm8:0
    instr |= cond << 8;
    return instr;
}

Instruction16 Base::b16(int16_t imm11) {
    Instruction16 instr = 0xe000;
    instr |= (0xfff & imm11) >> 1; // set imm11 (imm32 = imm11:0)
    return instr;
}

// imm32 = S:J2:J1:imm6:imm11:0
Instruction32 Base::bCond32(Condition cond, int32_t label) {
    if (cond == AL) {
        return b32(label);
    }
    Instruction32 instr = 0xf000'8000;
    instr |= cond << 22;
    label >>= 1; // left shift of 1 is applied by the instruction
    instr |= (0x7ff & label); // set imm11
    instr |= (0x3f & (label >> 11)) << 16; // imm6
    instr |= (0x1 & (label >> 17)) << 13; // J1
    instr |= (0x1 & (label >> 18)) << 11; // J2
    instr |= (0x1 & (label >> 19)) << 26; // J2
    return instr;
}

// imm32 = S:I2:I1:imm10:imm11:0
Instruction32 Base::b32(uint32_t label) {
    return Base::nop32();
}

Instruction16 Base::udf(uint8_t imm8) {
    Instruction16 instr = 0xde00;
    instr |= imm8;
    return instr;
}

// C1.5 Modified immediate constants, p. 473 
bool Base::canEncodeImmediateConstant(uint32_t constant) {
    // imm 0000x
    // const 0000000 00000000 00000000 abcdefgh
    if ((0xffff'ff00 & constant) == 0) {
        return true;
    }

    // imm 0001x
    // const 00000000 abcdefgh 00000000 abcdefgh
    if ((0xff00'ff00 & constant) == 0 && (0xff & (constant >> 16)) == (0xff & constant)) {
        return true;
    }

    // imm 0010x
    // const abcdefgh 00000000 abcdefgh 00000000
    if ((0x00ff'00ff & constant) == 0 && (0x0000'ff00 & (constant >> 16)) == (0x0000'ff00 & constant)) {
        return true;
    }

    // imm 0011x
    // const abcdefgh abcdefgh abcdefgh abcdefgh
    if (((0xff) & constant) == (0xff & (constant >> 8)) && ((0xff) & constant) == (0xff & (constant >> 16)) && ((0xff) & constant) == (0xff & (constant >> 24))) {
        return true;
    }

    // const 00...1bcdefgh0000....
    for (uint32_t i = 31; i >= 8; i--) {
        if ((constant >> i) == 1 && (0x00ff'ffff & (constant << (31-i))) == 0) {
            return true;
        }
    }

    return false;
}

Instruction32 Base::encodeImmediateConstant(Instruction32 instr, uint32_t constant) {
    // imm 0000x
    // const 0000000 00000000 00000000 abcdefgh
    if ((0xffff'ff00 & constant) == 0) {
        instr |= 0xff & constant; // imm8
        instr |= 0 << 26; // i
        instr |= 0x0 << 12; // imm3
        return instr;
    }

    // imm 0001x
    // const 00000000 abcdefgh 00000000 abcdefgh
    if ((0xff00'ff00 & constant) == 0 && (0xff & (constant >> 16)) == (0xff & constant)) {
        instr |= 0xff & constant; // imm8
        instr |= 0 << 26; // i
        instr |= 0x1 << 12; // imm3
        return instr;
    }

    // imm 0010x
    // const abcdefgh 00000000 abcdefgh 00000000
    if ((0x00ff'00ff & constant) == 0 && (0x0000'ff00 & (constant >> 16)) == (0x0000'ff00 & constant)) {
        instr |= 0xff & (constant >> 8); // imm8
        instr |= 0 << 26; // i
        instr |= 0x2 << 12; // imm3
        return instr;
    }

    // imm 0011x
    // const abcdefgh abcdefgh abcdefgh abcdefgh
    if (((0xff) & constant) == (0xff & (constant >> 8)) && ((0xff) & constant) == (0xff & (constant >> 16)) && ((0xff) & constant) == (0xff & (constant >> 24))) {
        instr |= 0xff & constant; // imm8
        instr |= 0 << 26U; // i
        instr |= 0x3 << 12U; // imm3
        return instr;
    }

    // const 00...1bcdefgh0000....
    for (uint32_t i = 31; i >= 8; i--) {
        if ((constant >> i) == 1 && (0x00ff'ffff & (constant << (31-i))) == 0) {
            // i:imm3:a for 31: 01000 = 8
            // i:imm3:a for 8: 11111 = 31
            // => 39-31 = 8, 39-30 = 9, ... 39-8 = 31
            uint8_t imm = 39 - i;
            instr |= (0x7 & (imm >> 1)) << 12; // imm3
            instr |= (0x1 & (imm >> 4)) << 26; // i
            instr |= (0x1 & imm) << 7; // a
            // for i = 31, shift 24; for i = 30, shift 23 ==> offset of 7
            instr |= 0x7f & (constant >> (i - 7));
            return instr;
        }
    }
    Base::printValidationError("encodeImmediateConstant: constant wasn't encoded - returning nop");
    return Base::nop32();
}

bool canEncodeMVEImmediateConstant(uint32_t constant) {
    return false;
}

Instruction32 Base::pldImmediate(Register Rn, uint16_t imm, bool write) {
    Instruction32 instr = 0xf890'f000;
    instr |= imm;
    instr |= Rn << 16;
    instr |= write << 21;
    return instr;
}
