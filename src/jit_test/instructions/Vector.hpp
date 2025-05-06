#ifndef JIT_INSTRUCTIONS_VECTOR_HPP
#define JIT_INSTRUCTIONS_VECTOR_HPP

#include "instructions/Base.hpp"
#include <cstdint>
namespace JIT {
    namespace Instructions {
        class Vector;
    }
}

class JIT::Instructions::Vector {
    public:
        /**
         * Floating Point Move between GP Register and Single-Precision Register
         * ARM V8M Reference C2.4.390, p. 1223
         * @param toGP If True moves data to the GP Register, else to the FP-Scalar Register
        */
        static Instruction32 vmovGPxScalar(bool toGP, FloatRegister Vn, Register Rt);

        static Instruction32 vmovImmediate(VectorRegister Qd, uint8_t imm8, DataType dt);
        static Instruction32 vmovRegister(VectorRegister Qd, VectorRegister Qm);

        static Instruction32 vldrw(VectorRegister Qd, Register Rn, uint8_t imm, bool preIndexed, bool writeBack, bool subtractImm);
        static Instruction32 vstrw(VectorRegister Qd, Register Rn, uint8_t imm, bool preIndexed, bool writeBack, bool subtractImm);


        static Instruction32 vfmaVectorByScalarPlusVector(VectorRegister Qda, VectorRegister Qn, Register Rm, bool bf16);
        static Instruction32 vfma(VectorRegister Qda, VectorRegister Qn, VectorRegister Qm, bool bf16);
};


#endif // JIT_INSTRUCTIONS_VECTOR_HPP