#include "Vector.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

using namespace JIT::Instructions;

// n = Vn:N
Instruction32 Vector::vmovGPxScalar(bool toGP, FloatRegister Vn, Register Rt) {
    Instruction32 instr = 0xEE000A10;

    instr |= static_cast<uint8_t>(toGP) << 20U;  // set op
    instr |= (Vn >> 1) << 16U; // Set Scalar Register
    instr |= (0x1 & Vn) << 7; // set N
    instr |= Rt << 12U; // Set GP Register
    
    return instr;
}

Instruction32 Vector::vldrw(VectorRegister Qd, Register Rn, int16_t imm, bool preIndexed, bool writeBack) {
    Instruction32 instr = 0xEC10'1F00;

    if (imm > 508 || imm < -508 || (imm & 0x03) != 0) {
        Base::printValidationError("vldrw: immediate must be +-[0, 508] and multiple of 4 - inserting nop");
        return Base::nop32();
    }
    if (!preIndexed && !writeBack) {
        Base::printValidationError("vldrw: post index must write back - setting write back");
        writeBack = true;
    }

    instr |= preIndexed << 24U; // Pre Indexed Variant (False -> Post-Indexed)
    instr |= writeBack << 21U;
    if (imm < 0) {
        imm = -imm;
    } else { // positive values: set A=1 == add immediate
        instr |= 1 << 23; // set subtract imm
    }

    instr |= (0x3ff & imm) >> 2; // mask off immediate and right shift because VLDRW does << 2
    instr |= Qd << 13U;
    instr |= Rn << 16U;

    return instr;
}

Instruction32 Vector::vstrw(VectorRegister Qd, Register Rn, int16_t imm, bool preIndexed, bool writeBack) {
    Instruction32 instr = vldrw(Qd, Rn, imm, preIndexed, writeBack);
    instr &= ~(1 << 20U); // 20th Bit is 0 instead of 1 (compared to vldrw) -> clear bit
    return instr;
}


JIT::Instructions::Instruction32 JIT::Instructions::Vector::vfmaVectorByScalarPlusVector(VectorRegister Qda, VectorRegister Qn, Register Rm, bool bf16) {
    Instruction32 instr = 0xEE31'0E40;

    instr |= bf16 << 28U; // use bf16 else use float32
    instr |= Qda << 13U;
    instr |= Qn << 17U;
    instr |= Rm;

    return instr;
}

Instruction32 Vector::vfma(VectorRegister Qda, VectorRegister Qn, VectorRegister Qm, bool bf16) {
    Instruction32 instr = 0xEF00'0C50;

    instr |= bf16 << 20U; // use bf16 else use float32
    instr |= Qm << 1U;
    instr |= Qda << 13U;
    instr |= Qn << 17U;

    return instr;
}

/*
* you have to use an cmode and op combination to generate the immediate
* an imm64 is generated, which is put into two lanes (when using single precision).
* this means, that the imm64 is written twice
* Combinations:
* - Int 32
*       cmode = 0000, op = 0
*           write imm8 in each lane
*       cmode = 0010, op = 0
*           write imm8<<8 in each lane
*       cmode = 0100, op = 0
*           write imm8<<16 in each lane
*       cmode = 0110, op = 0
*           write imm8<<24 in each lane
*       cmode = 1100, op = 0
*           write imm8:0xff in each lane
*       cmode = 1101, op = 0
*           write imm8:0xffff in each lane
* - Int 16
*       cmode = 1000, op = 0
*           write imm8 in each lane
*       cmode = 1010, op = 0
*           write imm8<<8 in each lane
* - Int 8
*       cmode = 1110, op = 0
*           write imm8 in each lane
* - Int 64
*       cmode = 1110, op = 1
*           replicate each bit of imm8 eight times and write the result to the two lanes
*           e.g. 01234567 -> 0000000011111111...
* - Float 32
*       cmode = 1111, op = 0
* imm8 = i:imm3:imm4
* for now only supports 8bit immediate
*/
Instruction32 Vector::vmovImmediate(VectorRegister Qd, uint8_t imm8, DataType dt) {
    Instruction32 instr = 0xef80'0050;
    instr |= Qd << 13;
    instr |= 0xf & imm8; // imm4
    instr |= (0x7 & (imm8 >> 4)) << 16; // imm3
    instr |= (0x1 & (imm8 >> 7)) << 28; // i

    if (dt == DataType::I8) {
        // only possible to write imm8 in each lane
        instr |= 0xe << 8; // cmode
        instr |= 0x0 << 5; // op
    } else if (dt == DataType::I16) {
        instr |= 0x8 << 8; // cmode
        instr |= 0x0 << 5; // op
    } else if (dt == DataType::I32) {
        instr |= 0x0 << 8; // cmode
        instr |= 0x0 << 5; // op
    } else {
        Base::printValidationError("vmovImmediate: Datatype not supported - returning nop");
        return Base::nop32();
    }
    return instr;
}

Instruction32 Vector::vorr(VectorRegister Qd, VectorRegister Qn, VectorRegister Qm) {
    Instruction32 instr = 0xef20'0150;
    instr |= Qm << 1;
    instr |= Qd << 13;
    instr |= Qn << 17;
    return instr;
}

Instruction32 Vector::vmovRegister(VectorRegister Qd, VectorRegister Qm) {
    return vorr(Qd, Qm, Qm); // vmov register is alias of vorr
}

Instruction32 Vector::vctp(Size size, Register Rn) {
    Instruction32 instr = 0xf000'e801;
    instr |= Rn << 16;
    instr |= size << 20;
    return instr;
}

// mask = Mkh:Mkl
Instruction32 Vector::vpst(uint8_t predicatedInstructions) {
    if (predicatedInstructions < 1 || predicatedInstructions > 4) {
        Base::printValidationError("vpst: only 1-4 instructions can be predicated - returning nop");
        return Base::nop32();
    }
    Instruction32 instr = 0xfe31'0f4d;
    switch (predicatedInstructions) {
        case 1:
            instr |= 1 << 22;
            break;
        case 2:
            instr |= 0b100 << 13;
            break;
        case 3:
            instr |= 0b010 << 13;
            break;
        case 4:
            instr |= 0b001 << 13;
            break;
    }
    return instr;
}