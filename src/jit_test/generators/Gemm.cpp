#include "Gemm.hpp"
#include "instructions/Arithmetic.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cstdint>

/**
 * @brief 
 * 
 * @param m 
 * @param k 
 * @param n 
 * @param lda 
 * @param ldb 
 * @param ldc 

 * Strategy:
 * - try to utilize an 8x3 microkernel (i.e. loop over 0-8*x for m (i loop) and 0-3*y for n (j loop)).
 * - if there are edge cases first handle the i loop cases 
 */
constexpr JIT::Instructions::VectorRegister C00_Register = JIT::Instructions::Q0;
constexpr JIT::Instructions::VectorRegister C01_Register = JIT::Instructions::Q1;
constexpr JIT::Instructions::VectorRegister C10_Register = JIT::Instructions::Q2;
constexpr JIT::Instructions::VectorRegister C11_Register = JIT::Instructions::Q3;
constexpr JIT::Instructions::VectorRegister C20_Register = JIT::Instructions::Q4;
constexpr JIT::Instructions::VectorRegister C21_Register = JIT::Instructions::Q5;
constexpr JIT::Instructions::VectorRegister A0_Register = JIT::Instructions::Q6;
constexpr JIT::Instructions::VectorRegister A1_Register = JIT::Instructions::Q7;
constexpr JIT::Instructions::Register B0_Register = JIT::Instructions::R7;
constexpr JIT::Instructions::Register B1_Register = JIT::Instructions::R8;
constexpr JIT::Instructions::Register B2_Register = JIT::Instructions::R9;
constexpr JIT::Instructions::Register I_Loop_Register = JIT::Instructions::R5;
constexpr JIT::Instructions::Register J_Loop_Register = JIT::Instructions::R4;
constexpr JIT::Instructions::Register A_Pointer = JIT::Instructions::R0;
constexpr JIT::Instructions::Register B_Pointer = JIT::Instructions::R1;
constexpr JIT::Instructions::Register C_Pointer = JIT::Instructions::R2;
constexpr JIT::Instructions::Register LDB_REGISTER = JIT::Instructions::R3;
constexpr JIT::Instructions::Register DLS_COUNT_REGISTER = JIT::Instructions::R6;
constexpr JIT::Instructions::Register A_Base_Pointer = JIT::Instructions::R12;
constexpr JIT::Instructions::Register C_ROW1_Base = JIT::Instructions::R10;
constexpr JIT::Instructions::Register C_ROW2_Base = JIT::Instructions::R11;

/* VLDRW uses 7bit immediate with LSL 2, i.e. 4byte aligned 9bit immediate */
constexpr uint32_t VLDR_TRESHOLD = 508;
constexpr uint32_t LDR_TRESHOLD = 4095;
/* Limit K unrolling to limit the buffer size */
constexpr uint32_t K_MAX_UNROLL = 5;
/* Use 8x3 microkernel by default */
constexpr uint32_t DEFAULT_MICROKERNEL_M = 16;
constexpr uint32_t DEFAULT_MICROKERNEL_N = 1;

constexpr uint32_t VECTOR_SIZE = 16; // == 128 Bit
constexpr uint32_t VECTOR_COUNT = 8;
constexpr uint32_t DT_SIZE = 4; // == 32 Bit (FP32)
constexpr uint32_t VECTOR_ELEMENTS = VECTOR_SIZE / DT_SIZE;

/*
Column-Major:
- lda: M
- ldb: K
- ldc: M

- VLDR: Used for C-Block and for A
    - For C: Supports ldc=M up to 508/4 = 127 for 8x2 and 508/8 = 63 for 8x3
    - For A: Supports lda=M up to 508/4 = 127

- LDR: Used for B
    - For B: Supports ldb=K up to 4095/4=1023 for 8x2 and 4095/8=511 for 8x3

- ADD: Used for Adding to C/A
    - Supports lda/ldc=M up to 4095/4=1024
*/
void JIT::Generators::Gemm::addImmediate(JIT::Instructions::Register reg, uint32_t immediate, Instructions::Register tempReg) {
    if (immediate <= LDR_TRESHOLD) {
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(reg, immediate));
    } else if (immediate <= 65535) {
        if (tempReg == Instructions::PC) {
            backend.addInstruction(Instructions::DataProcessing::push16(Instructions::R0));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R0, immediate));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(reg, Instructions::R0));
            backend.addInstruction(Instructions::DataProcessing::pop16(Instructions::R0));
        } else {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(tempReg, immediate));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(reg, tempReg));
        }
    } else {
        if (tempReg == Instructions::PC) {
            backend.addInstruction(Instructions::DataProcessing::push16(Instructions::R0));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R0, immediate));
            backend.addInstruction(Instructions::DataProcessing::movtImmediate32(Instructions::R0, immediate>>16));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(reg, Instructions::R0));
            backend.addInstruction(Instructions::DataProcessing::pop16(Instructions::R0));
        }
    }
}

void JIT::Generators::Gemm::generateMicroKernel(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, bool rewind) {    
    // calculate needed vector registers
    uint32_t neededVectorRegisters = m % VECTOR_ELEMENTS != 0 ? ((m / VECTOR_ELEMENTS) + 1) * n + ((m / VECTOR_ELEMENTS) + 1) : (m / VECTOR_ELEMENTS) * n + (m / VECTOR_ELEMENTS);
    uint32_t lastVldrImm = 0;
    // if not all elements fit into a single vector register, the instructions have to be predicated
    // the helium vector register can hold 4 fp32 values
    bool predicated = m % VECTOR_ELEMENTS != 0;
    uint32_t maxSkippedAdds = VLDR_TRESHOLD / (DT_SIZE * lda + (m-4) * DT_SIZE);
    maxSkippedAdds = maxSkippedAdds > K_MAX_UNROLL ? K_MAX_UNROLL : maxSkippedAdds; // limit k unrolling

    if (neededVectorRegisters > VECTOR_COUNT) {
        Instructions::Base::printValidationError("generateMicroKernel: dimensions too high - cant fit in vector registers");
        backend.addInstruction(Instructions::Base::nop32());
        return;
    }
    if (m <= 4) {
        bool useImmediateRow1 = DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;
        bool useImmediateRow2 = 2 * DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;
        bool useImmediateRow3 = 3 * DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;
        bool useImmediateRow4 = 4 * DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;
        bool useImmediateRow5 = 5 * DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;
        bool useImmediateRow6 = 6 * DT_SIZE * ldc + VECTOR_ELEMENTS * DT_SIZE <= VLDR_TRESHOLD;

        uint32_t vldrImmA = 0;
        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k-2) / maxSkippedAdds));

        if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 2 * DT_SIZE * ldb)); // TODO use immediate (even though ldb <= 500 is supported). important as ldb = K for column major
        if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, DT_SIZE * ldb));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));


        if (!useImmediateRow1 && n >= 2) backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_ROW1_Base, C_Pointer, DT_SIZE * ldc));
        if (!useImmediateRow2 && n == 3) backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_ROW2_Base, C_Pointer, 2 * DT_SIZE * ldc));

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        backend.addInstruction(Instructions::Vector::vldrw(C00_Register, C_Pointer));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        if (n >= 2) {
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vldrw(C10_Register, C_Pointer, 4 * ldc));
            else backend.addInstruction(Instructions::Vector::vldrw(C10_Register, C_ROW1_Base));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
        }
        if (n >= 3) {
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vldrw(C20_Register, C_Pointer, 8 * ldc));
            else backend.addInstruction(Instructions::Vector::vldrw(C20_Register, C_ROW2_Base));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
        }

        vldrImmA = lda * DT_SIZE;
        if (lda * DT_SIZE > VLDR_TRESHOLD) { // check if immediate can be used (only needed for really large sizes)
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * DT_SIZE));
            vldrImmA = 0;
        }

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA));
        vldrImmA = 0;
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));

        if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb * 4));
        if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb * 8));

        Instructions::Instruction16 * kLoopStart;
        if (k > 3) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
                if (i == 0) {
                    if (n == 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                    else {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                    }
                } else {
                    if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                }
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, LDB_REGISTER, 3));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * 4));
                if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, LDB_REGISTER, 2));
            }
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, maxSkippedAdds * lda * 4));
        }
        if (k > 3) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop
        for (uint32_t i = 0; i < (k-2) % maxSkippedAdds; i++) {
            if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            // add A_Pointer, i*lda
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * DT_SIZE));
            if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, LDB_REGISTER, 2));
            if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, LDB_REGISTER, 3));
        }
        // only add immediate if needed (TODO: is never needed and we can just use immediate in the next vldrw instructions)
        if ((k-2) % maxSkippedAdds > 0) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ((k-2) % maxSkippedAdds) * lda * 4));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        if (predicated) { // predicate next 3 instructions
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % 4)); // TODO move a bit earlier maybe (?)
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(2));
        }
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
        if (predicated && n >= 2) { // predicate the next rows. only needed if the rows are used (i.e. for 4x2, 4x3 microkernel)
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(n == 3 ? 4 : 2));
        }
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc * DT_SIZE));
            else backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_ROW1_Base));
        }
        if (n == 3) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc * DT_SIZE * 2));
            else backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_ROW2_Base));
        }
    } else if (m <= 8) {
        /* Load B */
        if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 8 * ldb)); // TODO use immediate (even though ldb <= 500 is supported). important as ldb = K for column major
        if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4 * ldb));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

        // determine if immediates can be used for loading/storing C
        // else we can use the extra registers for storing the pointers to the rows
        bool useImmediateRow1 = 4 * ldc + 16 <= VLDR_TRESHOLD;// && n >= 2;
        bool useImmediateRow2 = 8 * ldc + 16 <= VLDR_TRESHOLD;// && n == 3;
        uint32_t vldrImmA = 0;
        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k-2) / maxSkippedAdds));

        if (!useImmediateRow1 && n >= 2) backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_ROW1_Base, C_Pointer, 4 * ldc));
        if (!useImmediateRow2 && n == 3) backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_ROW2_Base, C_Pointer, 8 * ldc));

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        backend.addInstruction(Instructions::Vector::vldrw(C00_Register, C_Pointer));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vldrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        if (n >= 2) {
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vldrw(C10_Register, C_Pointer, 4 * ldc));
            else backend.addInstruction(Instructions::Vector::vldrw(C10_Register, C_ROW1_Base));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vldrw(C11_Register, C_Pointer, 4 * ldc + 16));
            else backend.addInstruction(Instructions::Vector::vldrw(C11_Register, C_ROW1_Base, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
        }
        if (n >= 3) {
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vldrw(C20_Register, C_Pointer, 8 * ldc));
            else backend.addInstruction(Instructions::Vector::vldrw(C20_Register, C_ROW2_Base));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vldrw(C21_Register, C_Pointer, 8 * ldc + 16));
            else backend.addInstruction(Instructions::Vector::vldrw(C21_Register, C_ROW2_Base, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
        }

        vldrImmA = lda * 4;
        if (lda * 4 > VLDR_TRESHOLD) { // check if immediate can be used (only needed for really large sizes)
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));
            vldrImmA = 0;
        }

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA));
        vldrImmA = 0;
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));

        if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb * 4));
        if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb * 8));

        Instructions::Instruction16 * kLoopStart;
        if (k > 3) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
                if (i == 0) {
                    if (n == 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                    else {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                    }
                } else {
                    if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                }
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));

                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, i * lda * 4 + 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * 4));
                if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
                if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, LDB_REGISTER, 2));
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
                if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, LDB_REGISTER, 3));
            }
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, maxSkippedAdds * lda * 4));
        }
        if (k > 3) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop
        for (uint32_t i = 0; i < (k-2) % maxSkippedAdds; i++) {
            if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            // add A_Pointer, i*lda
            backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, i * lda * 4 + 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * 4));
            if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            if (n >= 2) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, LDB_REGISTER, 2));
            if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            if (n == 3) backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, LDB_REGISTER, 3));
        }
        // only add immediate if needed (TODO: is never needed and we can just use immediate in the next vldrw instructions)
        if ((k-2) % maxSkippedAdds > 0) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ((k-2) % maxSkippedAdds) * lda * 4));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        if (predicated) { // predicate next 3 instructions
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % 4)); // TODO move a bit earlier maybe (?)
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc * 4));
            else backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_ROW1_Base));
        }
        if (predicated && n >= 2) { // predicate the next rows. only needed if the rows are used (i.e. for 8x2, 8x3 microkernel)
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(n == 3 ? 4 : 2));
        }
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            if (useImmediateRow1) backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_Pointer, ldc * 4 + 16));
            else backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_ROW1_Base, 16));
        }
        if (n == 3) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vstrw(C21_Register, C_Pointer, ldc * 8 + 16));
            else backend.addInstruction(Instructions::Vector::vstrw(C21_Register, C_ROW2_Base, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            if (useImmediateRow2) backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc * 8));
            else backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_ROW2_Base));
        }
    } else {
        // initialize accumulators and calculate first iteration of k

        /**
         * Used registers:
         *  - C: Q0-Q3
         *  - A: Q4-Q7
         * 
         */
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        uint32_t vldrImmA = 0;
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k - 2) / maxSkippedAdds));

        /* First Iteration */
        for (uint32_t i = 0; i < m; i += 4) {
            Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(i / 4);
            Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(i / 4 + 4);

            backend.addInstruction(Instructions::Vector::vldrw(cReg, C_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
        }

        /*vldrImmA = lda * 4;
        if (vldrImmA > VLDR_TRESHOLD) {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, vldrImmA));
            vldrImmA = 0;
        }*/

        //backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q4, A_Pointer, vldrImmA));
        //vldrImmA = 0;
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));

        Instructions::Instruction16 * kLoopStart;
        if (k > 3) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
                if (i == 0) {
                    kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                } else {
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                }
                for (uint32_t j = 0; j < m; j += 4) {
                    Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(j / 4);
                    Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(j / 4 + 4);

                    backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, i * lda * DT_SIZE + j * DT_SIZE));
                    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
                }
            }
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, maxSkippedAdds * lda * DT_SIZE));
        }
        if (k > 3) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop
        for (uint32_t i = 0; i < (k-2) % maxSkippedAdds; i++) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            for (uint32_t j = 0; j < m; j += 4) {
                Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(j / 4);
                Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(j / 4 + 4);

                backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer,  i * lda * 4 + j * 4));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
            }
        }
        // only add immediate if needed (TODO: is never needed and we can just use immediate in the next vldrw instructions)
        if ((k-2) % maxSkippedAdds > 0) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ((k-2) % maxSkippedAdds) * lda * 4));

        /* Last Iteration */
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        
        if (predicated) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % 4)); // TODO move a bit earlier maybe (?)
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        for (uint32_t i = m / 4; i > 0; i -= 1) {
            Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(i - 1);
            Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(i - 1 + 4);

            backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, (i-1) * 4 * 4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(cReg, C_Pointer, (i-1) * 4 * 4));
        }
    }
}


void (*JIT::Generators::Gemm::generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc)) (float const * a, float const * b, float * c) {
    backend.resetKernel();

    /*
     * INIT:
     * - Push callee saved registers and vector registers
     * - save base pointers
     * - initialize j counter
     */
    backend.addInstruction(JIT::Instructions::DataProcessing::push32(Instructions::R4, Instructions::R5, DLS_COUNT_REGISTER, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::LR));
    backend.addInstruction(JIT::Instructions::DataProcessing::vpush(Instructions::Q4, 4));
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(LDB_REGISTER, ldb));

    // can be solved by using a single microkernel (no loops required)
    // - 16x1
    // - 8x2, 8x3
    // - 4x4-4x7
    if ((m <= 16 && n == 1) || (m <= 8 && n <= 3) || (m <= 4 && n <= 7)) {
        generateMicroKernel(m, k, n, lda, ldb, ldc);
    } else if (n == 1) { // dont need j=n loop (only i loop) and use wide microkernel // TODO: use DEFAULT_MICROKERNEL_N and then select widest possible microkernel
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));
        Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Base::nop32());

        generateMicroKernel(16, k, 1, lda, ldb, ldc);
        
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 16));

        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
        // Rewind C => C += 8*4
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 16 * DT_SIZE));

        // ensure that only full microkernels can be executed
        backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % 16)));
        backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

        if (m % 16 != 0) {
            generateMicroKernel(m % 16, k, 1, lda, ldb, ldc);
        }
    } else if (m == 1) { // dont need i=m loop (only j loop)

    } else {
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(J_Loop_Register, 0)); // mov 0 to r4

        /*
        * Loop j (n loop): Count from 0 to n
        */
        Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));

        /*
        * Loop i (m loop): Count from 0 to m
        */
        Instructions::Instruction16 * iLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32());

        generateMicroKernel(DEFAULT_MICROKERNEL_M, k, DEFAULT_MICROKERNEL_N, lda, ldb, ldc);

        // prepare for next i loop. execute earlier to access new i value
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M));

        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        // b will be incremented in each k-Iteration by ldr r7, [r1], #4. this is done k times and needs to be reverted
        backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
        // Forward C => C += 8*4
        // After each block is computed in the m-loop, C has to move forward to the next block
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));

        // ensure that only full microkernels can be executed
        backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % DEFAULT_MICROKERNEL_M)));
        backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT);

        // handle i loop edge cases
        if (m % DEFAULT_MICROKERNEL_M != 0) {
            generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, DEFAULT_MICROKERNEL_N, lda, ldb, ldc);

            // prepare for next i loop. execute earlier to access new i value
            // backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));
            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            // Rewind B => B = B - 4*k
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
            // Rewind C => C += 8*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (m % DEFAULT_MICROKERNEL_M) * DT_SIZE));
        }

        // gemm loop i end (next j)
        // Rewind A -> already rewinded by i so need to reset
        backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B -> rewinded by i to start. add 3*len
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, ldb * DEFAULT_MICROKERNEL_N * DT_SIZE));
        // Rewind C -> still have to go two lines -> 2ldc
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (DEFAULT_MICROKERNEL_N - 1) * ldc * DT_SIZE));

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(J_Loop_Register, DEFAULT_MICROKERNEL_N));
        backend.addInstruction(Instructions::Base::cmpImmediate16(J_Loop_Register, n - (n % DEFAULT_MICROKERNEL_N)));
        backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);

        // handle j loop edge cases
        if (n % DEFAULT_MICROKERNEL_N != 0) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));
            Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Base::nop32());

            generateMicroKernel(DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc);
            
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M));

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
            // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
            // Rewind B => B = B - 4*k
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4 * k));
            // Rewind C => C += 8*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, DEFAULT_MICROKERNEL_M * 4));

            // ensure that only full microkernels can be executed
            backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % DEFAULT_MICROKERNEL_M)));
            backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

            if (m % DEFAULT_MICROKERNEL_M != 0) {
                generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc);
            }
        }
    }

    // gemm loop j end
    backend.addInstruction(Instructions::DataProcessing::vpop(Instructions::Q4, 4));
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4, Instructions::R5, DLS_COUNT_REGISTER, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::PC));

    __asm("dsb");
    __asm("isb");
    return reinterpret_cast<Func>(backend.getThumbAddress());
}
