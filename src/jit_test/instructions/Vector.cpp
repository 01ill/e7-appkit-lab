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
    Instruction32 instr = 0xEC10'1F00;

    // the combination of post indexed and not writing back is not allowed
    // it will be handled as a default case of just loading without any offset i.e. pre-indexed with zero-immediate
    if (!preIndexed && !writeBack) {
        imm = 0;
        preIndexed = 1;
    }

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
    Instruction32 instr = 0xEE31'0E40;

    instr |= bf16 << 28U; // use bf16 else use float32
    instr |= Qda << 13U;
    instr |= Qn << 17U;
    instr |= Rm;

    return instr;
}

JIT::Instructions::Instruction32 JIT::Instructions::Vector::vfma(VectorRegister Qda, VectorRegister Qn, VectorRegister Qm, bool bf16) {
    Instruction32 instr = 0xEF00'0C50;

    instr |= bf16 << 20U; // use bf16 else use float32
    instr |= Qm << 1U;
    instr |= Qda << 13U;
    instr |= Qn << 17U;

    return instr;
}
