#include "Vector.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

JIT::Instructions::Instruction32 JIT::Instructions::Vector::vmovGPxScalar(bool toGP, FloatRegister Vn, Register Rt) {
    Instruction32 instr = 0xEE000A10;

    instr |= static_cast<uint8_t>(toGP) << 20U;  // set op
    instr |= Vn << 16U; // Set Scalar Register
    instr |= Rt << 12U; // Set GP Register
    
    return instr;
}

JIT::Instructions::Instruction32 JIT::Instructions::Vector::vldrw(VectorRegister Qd, Register Rn, uint8_t imm, bool preIndexed, bool writeBack, bool subtractImm) {
    Instruction32 instr = 0xEC101F00;

    instr |= preIndexed << 24U; // Pre Indexed Variant (False -> Post-Indexed)
    instr |= writeBack << 21U;
    instr |= !subtractImm << 23U;
    instr |= 0x7f & imm; // mask off highest bit
    instr |= Qd << 13U;
    instr |= Rn << 16U;

    return instr;
}

JIT::Instructions::Instruction32 JIT::Instructions::Vector::vstrw(VectorRegister Qd, Register Rn, uint8_t imm, bool preIndexed, bool writeBack, bool subtractImm) {
    Instruction32 instr = vldrw(Qd, Rn, imm, preIndexed, writeBack, subtractImm);
    instr &= ~(1 << 20U); // 20th Bit is 0 instead of 1 (compared to vldrw) -> clear bit
    return instr;
}


JIT::Instructions::Instruction32 JIT::Instructions::Vector::vfmaVectorByScalarPlusVector(VectorRegister Qda, VectorRegister Qn, Register Rm, bool bf16) {
    Instruction32 instr = 0xEE310E40;

    instr |= bf16 << 28U; // use bf16 else use float32
    instr |= Qda << 13U;
    instr |= Qn << 17U;
    instr |= Rm;

    return instr;
}
