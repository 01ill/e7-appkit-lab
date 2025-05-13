#include "Arithmetic.hpp"
#include "instructions/Base.hpp"
#include <cassert>
#include <cstdint>

using namespace JIT::Instructions;

Instruction16 Arithmetic::addImmediate16(Register Rd, Register Rn, uint8_t imm3) {
    #ifdef VALIDATE_ENCODINGS
    assert(Rn <= Register::R7);
    assert(Rd <= Register::R7);
    assert(imm3 <= 7);
    #endif

    Instruction16 instr = 0x1c00;
    instr |= Rd; // set Rd register
    instr |= Rn << 3U;
    instr |= (0x07 & imm3) << 6U; // set imm3 and ensure 3 bits
    return instr;
}

Instruction16 Arithmetic::addImmediate16(Register Rdn, uint8_t imm8) {
    #ifdef VALIDATE_ENCODINGS
    assert(Rdn <= Register::R7);
    #endif
    Instruction16 instr = 0x3000;
    instr |= imm8; // set imm8
    instr |= Rdn << 8U; // set Rdn register
    return instr;
}

// imm12 = i:imm3:imm8
// T3: encode 32bit with modified immediate constant (ignored for now -> use only T4)
Instruction32 Arithmetic::addImmediate32(Register Rd, Register Rn, uint16_t imm12, bool setFlags) {
    #ifdef VALIDATE_ENCODINGS
    assert(imm12 <= 4095);
    #endif
    
    Instruction32 instr = 0xf200'0000;
    instr |= Rd << 8U; // set Rd
    instr |= Rn << 16U; // set Rn

    imm12 &= 0x0fff; // ensure 12bit
    instr |= 0x0ff & imm12; // set imm8

    instr |= (0x7 & (imm12 >> 8U)) << 12U; // set imm3
    instr |= (imm12 >> 11U) << 26U; // set i
    // instr |= setFlags << 20U; // set S (currently ignored because we use the T4 encoding)
    return instr;
}

Instruction32 Arithmetic::addImmediate32(Register Rd, uint16_t imm12, bool setFlags) {
    return addImmediate32(Rd, Rd, imm12, setFlags);
}

Instruction16 Arithmetic::addRegister16(Register Rd, Register Rn, Register Rm) {
    Instruction16 instr = 0x1800;

    instr |= Rd;
    instr |= Rn << 3U;
    instr |= Rm << 6U;
    return instr;
}

Instruction16 Arithmetic::addRegister16(Register Rdn, Register Rm) {
    Instruction16 instr = 0x4400;

    instr |= 0x7 & Rdn; // set Rdn (3 lowest bits)
    instr |= (Rdn >> 3U) << 7U; // set DN (highest bit from Rdn)
    instr |= Rm << 3U;
    return instr;
}

// amount = imm3:imm2
Instruction32 Arithmetic::addRegister32(Register Rd, Register Rn, Register Rm, Shift shift, uint8_t amount, bool setFlags) {
    Instruction32 instr = 0xeb00'0000;

    instr |= Rm;
    instr |= Rd << 8U;
    instr |= Rn << 16U;
    instr |= shift << 4;
    amount &= 0x1f; // imm5
    instr |= (0x3 & amount) << 6; // set imm2
    instr |= (amount >> 2) << 12;
    instr |= setFlags << 20;
    return instr;
}

Instruction32 Arithmetic::addRegister32(Register Rd, Register Rm, Shift shift, uint8_t amount, bool setFlags) {
    return addRegister32(Rd, Rd, Rm, shift, amount, setFlags);
}


Instruction16 Arithmetic::subImmediate16(Register Rd, Register Rn, uint8_t imm3) {
    if (!Base::assertLowRegister(Rd, Rn)) {
        Base::printValidationError("subImmediate16: Only low registers allowed - returning nop");
        return Base::nop16();
    }
    if (imm3 > 7) {
        Base::printValidationError("subImmediate16: imm3 must be <7 - returning nop");
        return Base::nop16();
    }
    Instruction16 instr = 0x1e00;
    instr |= Rd;
    instr |= Rn << 3;
    instr |= (0x7 & imm3) << 6;
    return instr;
}

Instruction16 Arithmetic::subImmediate16(Register Rdn, uint8_t imm8) {
    if (!Base::assertLowRegister(Rdn)) {
        Base::printValidationError("subImmediate16: Only low registers allowed - returning nop");
        return Base::nop16();
    }
    Instruction16 instr = 0x3800;
    instr |= imm8;
    instr |= Rdn << 8;
    return instr;
}

// imm12 = i:imm3:imm8
Instruction32 Arithmetic::subImmediate32(Register Rd, Register Rn, uint16_t imm12) {
    if (imm12 > 4095) {
        Base::printValidationError("subImmediate32: imm12 must be <4095 - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xf2a0'0000;
    instr |= 0xff & imm12; // set imm8
    instr |= (0x7 & (imm12 >> 8)) << 12; // set imm3
    instr |= (0x1 & (imm12 >> 11)) << 26; // set i
    instr |= Rd << 8;
    instr |= Rn << 16;
    return instr;
}

Instruction32 Arithmetic::subImmediate32(Register Rdn, uint16_t imm12) {
    return subImmediate32(Rdn, Rdn, imm12);
}


Instruction16 Arithmetic::mul16(Register Rdm, Register Rn) {
    if (!Base::assertLowRegister(Rdm, Rn)) {
        Base::printValidationError("mul16: Only low registers allowed - returning nop");
        return Base::nop16();
    }
    Instruction16 instr = 0x4340;
    instr |= Rdm;
    instr |= Rn << 3;
    return instr;
}

Instruction32 Arithmetic::mul32(Register Rd, Register Rn, Register Rm) {
    Instruction32 instr = 0xfb00'f000;
    instr |= Rm;
    instr |= Rd << 8;
    instr |= Rn << 16;
    return instr;
}
