#include "Gemm.hpp"
#include "SEGGER_RTT.h"
#include "backend/Backend.hpp"
#include "instructions/Arithmetic.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cstdint>


constexpr JIT::Instructions::VectorRegister A0_Register = JIT::Instructions::Q6;
constexpr JIT::Instructions::VectorRegister A1_Register = JIT::Instructions::Q7;
constexpr JIT::Instructions::VectorRegister C00_Register = JIT::Instructions::Q0;
constexpr JIT::Instructions::VectorRegister C01_Register = JIT::Instructions::Q1;
constexpr JIT::Instructions::VectorRegister C10_Register = JIT::Instructions::Q2;
constexpr JIT::Instructions::VectorRegister C11_Register = JIT::Instructions::Q3;
constexpr JIT::Instructions::VectorRegister C20_Register = JIT::Instructions::Q4;
constexpr JIT::Instructions::VectorRegister C21_Register = JIT::Instructions::Q5;
/* When generating a 4x4-4x6 microkernel the Cx1 registers can be used */
constexpr JIT::Instructions::VectorRegister C30_Register = C01_Register;
constexpr JIT::Instructions::VectorRegister C40_Register = C11_Register;
constexpr JIT::Instructions::VectorRegister C50_Register = C21_Register;
constexpr JIT::Instructions::Register B0_Register = JIT::Instructions::R8;
constexpr JIT::Instructions::Register B1_Register = JIT::Instructions::R7;
constexpr JIT::Instructions::Register B2_Register = JIT::Instructions::R6;
constexpr JIT::Instructions::Register I_Loop_Register = JIT::Instructions::R5;
constexpr JIT::Instructions::Register J_Loop_Register = JIT::Instructions::R4;
constexpr JIT::Instructions::Register A_Pointer = JIT::Instructions::R0;
constexpr JIT::Instructions::Register B_Pointer = JIT::Instructions::R1;
constexpr JIT::Instructions::Register C_Pointer = JIT::Instructions::R2;
constexpr JIT::Instructions::Register DLS_COUNT_REGISTER = JIT::Instructions::R9;
constexpr JIT::Instructions::Register A_Base_Pointer = JIT::Instructions::R3;
/* The LENx registers are assigned when determining the microkernel strategy */
constexpr JIT::Instructions::Register LEN1_REGISTER = JIT::Instructions::R10;
constexpr JIT::Instructions::Register LEN2_REGISTER = JIT::Instructions::R11;
constexpr JIT::Instructions::Register LEN3_REGISTER = JIT::Instructions::R12;
 
/* VLDRW uses 7bit immediate with LSL 2, i.e. 4byte aligned 9bit immediate */
constexpr uint32_t VLDR_TRESHOLD = 508;
/* LDR/ADD/SUB use 12bit immediate, i.e. offset of 4095 allowed*/
constexpr uint32_t LDR_TRESHOLD = 4095;
/* MOV uses 16bit immediate */
constexpr uint32_t MOV_TRESHOLD = 65535;
/* Limit K unrolling to limit the buffer size
- 24x24x24: K=1: 1.457
            K=2: 1.457
            K=3: 1.46
            K=10: 1.46
*/
constexpr uint32_t K_MAX_UNROLL = 5;
constexpr uint32_t M_MAX_UNROLL = 5;
constexpr uint32_t N_MAX_UNROLL = 5;
/* Use 8x3 microkernel by default */
constexpr uint32_t DEFAULT_MICROKERNEL_M = 8;
constexpr uint32_t DEFAULT_MICROKERNEL_N = 3;

constexpr uint32_t VECTOR_SIZE = 16; // == 128 Bit
/* Count of vector registers */
constexpr uint32_t VECTOR_COUNT = 8;
constexpr uint32_t DT_SIZE = 4; // == 32 Bit (FP32)
/* Calculate elements of elements which fit into a single vector register */
constexpr uint32_t VECTOR_ELEMENTS = VECTOR_SIZE / DT_SIZE;

void JIT::Generators::Gemm::emitLoadB(JIT::Instructions::Register targetReg, MicroKernelConfiguration & configuration, uint32_t leftShiftAmount, uint32_t offset, bool secondHalf, bool try16Bit) {
    if ((configuration.registerStrategy & (secondHalf ? USE_BCOL3_REGISTER : USE_LDB_REGISTER)) && (targetReg == B1_Register || targetReg == B2_Register)) {
        backend.addInstruction(Instructions::DataProcessing::ldrRegister32(targetReg, secondHalf ? configuration.BCOL3_REGISTER : B_Pointer, configuration.LDB_REGISTER, leftShiftAmount));
    } else if (targetReg == B1_Register || targetReg == B2_Register) {
        if (try16Bit && Instructions::Base::assertLowRegister(targetReg) && offset <= 124 && offset % 4 == 0) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer , offset));
        } else if (offset <= LDR_TRESHOLD) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer , offset));
        } else {
            // no fallback is provided as it is absolutely essential for performance that no extra instructions are wasted for b loads
            Instructions::Base::printValidationError("emitLoadB Immediate Fallback; inserting nop");
            backend.addInstruction(Instructions::Base::nop32());
        }
    } else { // if targetReg == B0_Register
        if (configuration.registerStrategy & USE_LDB_REGISTER && !secondHalf) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer, DT_SIZE, false, true));
        } else if (configuration.registerStrategy & USE_BCOL3_REGISTER && secondHalf) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, configuration.BCOL3_REGISTER, DT_SIZE, false, true));
        } else if (secondHalf) {
            if (offset <= LDR_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer, offset));
            } else {
                // no fallback is provided as it is absolutely essential for performance that no extra instructions are wasted for b loads
                Instructions::Base::printValidationError("emitLoadB Immediate Fallback; inserting nop");
                backend.addInstruction(Instructions::Base::nop32());
            }
        }
    }
}

void JIT::Generators::Gemm::emitLoadStoreC(MicroKernelConfiguration & configuration, Instructions::VectorRegister targetReg, uint32_t ldc, bool store) {
    #ifdef FLAG_NO_C_LOAD
    if (!store) {
        backend.addInstruction(Instructions::Vector::vmovImmediate(targetReg, 0, Instructions::I32));
        return;
    }
    #endif
    if (targetReg == C00_Register || targetReg == C01_Register) {
        if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, C_Pointer, targetReg == C01_Register ? 16 : 0));
        else backend.addInstruction(Instructions::Vector::vldrw(targetReg, C_Pointer, targetReg == C01_Register ? 16 : 0));
    } else {
        bool secondRow = targetReg == C20_Register || targetReg == C21_Register;
        bool rightSide = targetReg == C11_Register || targetReg == C21_Register;
        if ((configuration.registerStrategy & USE_CROW2_REGISTER && secondRow) || (configuration.registerStrategy & USE_CROW1_REGISTER && !secondRow)) {
            if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, secondRow ? configuration.CROW2_REGISTER : configuration.CROW1_REGISTER, rightSide ? 16 : 0));
            else backend.addInstruction(Instructions::Vector::vldrw(targetReg, secondRow ? configuration.CROW2_REGISTER : configuration.CROW1_REGISTER, rightSide ? 16 : 0));
        } else {
            uint32_t offset = secondRow ? 2 * DT_SIZE * ldc : DT_SIZE * ldc;
            if (rightSide) offset += 4 * DT_SIZE;
            if (offset <= VLDR_TRESHOLD) {
                if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, C_Pointer, offset));
                else backend.addInstruction(Instructions::Vector::vldrw(targetReg, C_Pointer, offset));
            } else {
                if (offset > LDR_TRESHOLD) {
                    if (offset > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, offset >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, offset));
                    backend.addInstruction(Instructions::Arithmetic::addRegister32(DLS_COUNT_REGISTER, C_Pointer));
                } else {
                    backend.addInstruction(Instructions::Arithmetic::addImmediate32(DLS_COUNT_REGISTER, C_Pointer, offset));
                }
                if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, DLS_COUNT_REGISTER));
                else backend.addInstruction(Instructions::Vector::vldrw(targetReg, DLS_COUNT_REGISTER));
            }
        }
    }

}

void JIT::Generators::Gemm::emitLoadStoreC46(Instructions::VectorRegister targetReg, uint32_t ldc, bool store) {
    uint32_t imm = ldc * DT_SIZE;
    if (imm > VLDR_TRESHOLD) {
        if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, C_Pointer));
        else backend.addInstruction(Instructions::Vector::vldrw(targetReg, C_Pointer));
        if (imm > LDR_TRESHOLD) {
            if (imm > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, imm >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, imm));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(C_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, imm));
        }
    } else {
        if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, C_Pointer, imm, false, true));
        else backend.addInstruction(Instructions::Vector::vldrw(targetReg, C_Pointer, imm, false, true));
    }
}

void JIT::Generators::Gemm::generateMicroKernel(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, MicroKernelConfiguration & configuration) {
    // calculate needed vector registers
    uint32_t neededVectorRegisters = m % VECTOR_ELEMENTS != 0 // check if predicates are needed
        ? ((m / VECTOR_ELEMENTS) + 1) * n + ((m / VECTOR_ELEMENTS) + 1) // with predicates
        : (m / VECTOR_ELEMENTS) * n + (m / VECTOR_ELEMENTS); // no predicates. all vector registers are fully used
    if (neededVectorRegisters > VECTOR_COUNT) {
        Instructions::Base::printValidationError("generateMicroKernel: dimensions too high - cant fit in vector registers");
        backend.addInstruction(Instructions::Base::nop32());
        return;
    }

    uint32_t kMiddle = k - 2; // k without first and last iteration
    // if the immediate for loading from A can't be encoded in VLDR we have to add to the pointer earlier
    bool aNeedsPreadd = lda * DT_SIZE > VLDR_TRESHOLD;
    // if not all elements fit into a single vector register, the instructions have to be predicated
    bool predicated = m % VECTOR_ELEMENTS != 0;
    // determine amount of k loop unrolling depending on the amount of possible skipped ADDs (possible if we can still encode next immediate with VLDR)
    uint32_t unrollK = VLDR_TRESHOLD / (DT_SIZE * lda);
    unrollK = unrollK > K_MAX_UNROLL ? K_MAX_UNROLL : unrollK; // limit k unrolling (priorize code size over (really small) performance gain)
    unrollK = unrollK > kMiddle ? kMiddle : unrollK; // limit unrolling if k is small
    unrollK = unrollK < 1 ? 1 : unrollK; // at least one iteration
    // we can omit the loop if we have only one iteration anyways (and k must be large enough; if k == 1, kMiddle == -1 == INT_MAX-1)
    bool needsDls = (kMiddle / unrollK) > 1 && k > 1;

    /* Path for 4x4-4x6 microkernel */
    if (m <= 4) {
        if (configuration.registerStrategy & USE_BCOL3_REGISTER) {
            uint32_t imm = 3 * ldb * DT_SIZE;
            if (imm > LDR_TRESHOLD) {
                if (imm > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.BCOL3_REGISTER, imm >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.BCOL3_REGISTER, imm));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(configuration.BCOL3_REGISTER, B_Pointer));
            } else backend.addInstruction(Instructions::Arithmetic::addImmediate32(configuration.BCOL3_REGISTER, B_Pointer, imm));
        }

        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
        backend.addHeliumInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer)); // load a[0]
        uint32_t vldrImmA = ldc * DT_SIZE;
        // if immediate doesn't fit into VLDR
        if (configuration.registerStrategy & USE_A_ADD_REGISTER) { //  we can use extra register with lda stored
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, configuration.A_ADD_REGISTER));
            vldrImmA = 0;
        } else if (aNeedsPreadd) { // no register available (wont normally be the case as A register is prioritised)
            if (vldrImmA < LDR_TRESHOLD) {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * DT_SIZE));
            } else {
                if (vldrImmA > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, vldrImmA >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, vldrImmA));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
            }
            vldrImmA = 0;
        }
        // load b[0] is emitted last in the first block of b load as it will perform a write back to step forward one row for the next k iteration
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true)); // load b[0]
        if (predicated && k == 1) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % VECTOR_ELEMENTS));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        emitLoadStoreC46(C00_Register, k > 1 ? ldc : 0); // load c[0][0-3]
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register)); // vfma c[0][0]
        if (k == 1) emitLoadStoreC46(C00_Register, ldc, true);
        if (n >= 2) {
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC46(C10_Register, k > 1 ? ldc : 0); // load c[1][0-3]
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register)); // vfma c[1][0...]
            if (k == 1) emitLoadStoreC46(C10_Register, ldc, true);
        }
        if (n >= 3) {
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC46(C20_Register, k > 1 ? ldc : 0); // load c[2][0...]
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register)); // vfma c[2][0...]
            if (k == 1) emitLoadStoreC46(C20_Register, ldc, true);
        }

        if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true); // load b[4ldb]
        if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true); // load b[5ldb]
        // b[3ldb] is emitted last in the second block of b loads as it will perform a write back
        if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true); // load b[3ldb]
        
        if (n >= 4) {
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC46(C30_Register, k > 1 ? ldc : 0);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register));
            if (k == 1) emitLoadStoreC46(C30_Register, ldc, true);
        }
        if (n >= 5) {
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC46(C40_Register, k > 1 ? ldc : 0);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register));
            if (k == 1) emitLoadStoreC46(C40_Register, ldc, true);
        }
        if (n >= 6) {
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC46(C50_Register, k > 1 ? ldc : 0);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register));
            if (k == 1) emitLoadStoreC46(C50_Register, ldc, true);
        }

        // early return for k == 1. only reset c pointer now
        if (k == 1) {
            if (n * ldc * DT_SIZE > LDR_TRESHOLD) {
                if (n * ldc * DT_SIZE > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (n * ldc * DT_SIZE) >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, n * ldc * DT_SIZE));
                backend.addInstruction(Instructions::Arithmetic::subRegister32(C_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::subImmediate32(C_Pointer, n * ldc * DT_SIZE));
            }
            return;
        }

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA)); // load a for next iteration
        vldrImmA = 0;
        if (!aNeedsPreadd) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * DT_SIZE));

        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        if (needsDls) {
            // handle edge case for really huge k
            if (kMiddle / unrollK > 65535) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (kMiddle / unrollK) >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, kMiddle / unrollK));
        }

        /* Load B */
        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]

        Instructions::Instruction16 * kLoopStart;
        if (needsDls) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < unrollK; i++) { // unroll k loop
                // first iteration of unrolling (for large M there will be no unrolling)
                // in the first iteration we have to set the starting point of the loop
                if (i == 0) {
                    // for 4x1 microkernel we have to omit the vfma c[1][0...]
                    if (n == 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true)); // 
                    else {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                    }
                } else { // in all other iterations no starting point is set
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                }
                if (n >= 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                if (n >= 2 && n < 4) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb] (different place for 4x3 microkernel to improve performance)
                if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true); // load b[4ldb]
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register)); // vfma with b[0]
                if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true); // load b[5ldb]
                if (n >= 5) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register)); // vfma with b[4ldb]
                if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true); // load b[3ldb]
                if (n >= 6) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register)); // vfma with b[5ldb]
                if (n >= 4) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb] (different order for 4x6)
                if (n >= 4) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register)); // vfma with b[3ldb]
                if (!aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * DT_SIZE)); // if we can still use immediates, load next A
                if (i < (unrollK - 1) && n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register)); // vfma with b[ldb]
                if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
            } // end unroll
            // before the next iteration the immediate is added to go to the next A column (or to the x next as we have skipped a few by unrolling)
            uint32_t skipAdds = unrollK * lda * DT_SIZE;
            if (skipAdds <= LDR_TRESHOLD) {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, skipAdds));
            } else {
                if (skipAdds > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, skipAdds >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, skipAdds));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
            }
            // if immediate doesn't fit place vldr for next A load at the end after adding the immediate (loop isn't unrolled then)
            if (aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        }
        if (needsDls) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop (in the same way as the k-loop)
        for (uint32_t i = 0; i < kMiddle % unrollK; i++) {
            if (i == 0 && n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
            if (n >= 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true);
            if (n >= 5) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register));
            if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true);
            if (n >= 6) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register));
            if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
            if (n >= 4) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register));
            if (!aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * DT_SIZE));
            if (i < ((kMiddle % unrollK)-1) && n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
        }

        if (predicated) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % VECTOR_ELEMENTS));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
        }

        // restore C Pointer
        if (n * ldc * DT_SIZE > LDR_TRESHOLD) {
            if (n * ldc * DT_SIZE > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (n * ldc * DT_SIZE) >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, n * ldc * DT_SIZE));
            backend.addInstruction(Instructions::Arithmetic::subRegister32(C_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(C_Pointer, n * ldc * DT_SIZE));
        }

        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));

        bool cNeedsPreadd = ldc * DT_SIZE > VLDR_TRESHOLD;
        if (predicated) backend.addInstruction(Instructions::Vector::vpst(2));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        emitLoadStoreC46(C00_Register, ldc, true);
        if (predicated && n >= 2) backend.addInstruction(Instructions::Vector::vpst(n >= 3 && !cNeedsPreadd  ? 4 : 2));
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            emitLoadStoreC46(C10_Register, ldc, true);
        }
        if (n >= 3) {
            if (predicated && cNeedsPreadd) backend.addInstruction(Instructions::Vector::vpst(2));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            emitLoadStoreC46(C20_Register, ldc, true);
        }
        
        if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true);
        if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true);
        if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true);

        if (predicated && n >= 4) backend.addInstruction(Instructions::Vector::vpst(n >= 5 && !cNeedsPreadd ? 4 : 2));
        if (n >= 4) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register));
            emitLoadStoreC46(C30_Register, ldc, true);
        }
        if (n >= 5) {
            if (predicated && cNeedsPreadd) backend.addInstruction(Instructions::Vector::vpst(2));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register));
            emitLoadStoreC46(C40_Register, ldc, true);
        }
        if (predicated && n >= 6) backend.addInstruction(Instructions::Vector::vpst(2));
        if (n >= 6) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register));
            emitLoadStoreC46(C50_Register, ldc, true);
        }

        // reset c pointer
        if (n * ldc * DT_SIZE > LDR_TRESHOLD) {
            if (n * ldc * DT_SIZE > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (n * ldc * DT_SIZE) >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, n * ldc * DT_SIZE));
            backend.addInstruction(Instructions::Arithmetic::subRegister32(C_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(C_Pointer, n * ldc * DT_SIZE));
        }
    /* path for 8x3 microkernel */
    } else if (m <= 8) {
        if (configuration.insertPreloadHints) backend.addInstruction(Instructions::Base::pldImmediate(A_Pointer, 0));
        // calculate pointers for second c row if needed
        if (configuration.registerStrategy & USE_CROW1_REGISTER) {
            uint32_t cRowAdd = DT_SIZE * ldc;
            if (cRowAdd > LDR_TRESHOLD) {
                if (cRowAdd > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, cRowAdd >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, cRowAdd));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(configuration.CROW1_REGISTER, C_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(configuration.CROW1_REGISTER, C_Pointer, cRowAdd));
            }
        }
        // calculate pointers for third c row if needed
        if (configuration.registerStrategy & USE_CROW2_REGISTER) {
            uint32_t cRowAdd = 2 * DT_SIZE * ldc;
            if (cRowAdd > LDR_TRESHOLD) {
                if (cRowAdd > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, cRowAdd >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, cRowAdd));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(configuration.CROW2_REGISTER, C_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(configuration.CROW2_REGISTER, C_Pointer, cRowAdd));
            }
        }
        if (configuration.insertPreloadHints) backend.addInstruction(Instructions::Base::pldImmediate(A_Pointer, 16));

        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        backend.addHeliumInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        // Add to A Pointer. If not large enough reuse DLS Count register
        // Load A[0]
        uint32_t vldrImmA = ldc * DT_SIZE;
        if (configuration.registerStrategy & USE_A_ADD_REGISTER) {
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, configuration.A_ADD_REGISTER));
            vldrImmA = 0;
        } else if (aNeedsPreadd) {
            if (vldrImmA < LDR_TRESHOLD) {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, vldrImmA));
                vldrImmA = 0;
            } else {
                if (vldrImmA > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, vldrImmA >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, vldrImmA));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
            }
            vldrImmA = 0;
        }
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        if (predicated && k == 1) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % VECTOR_ELEMENTS));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        backend.addInstruction(Instructions::Vector::vldrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        if (k == 1) emitLoadStoreC(configuration, C01_Register, ldc, true);
        if (n >= 2) {
            emitLoadStoreC(configuration, C10_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            if (k == 1) emitLoadStoreC(configuration, C10_Register, ldc, true);
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC(configuration, C11_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            if (k == 1) emitLoadStoreC(configuration, C11_Register, ldc, true);
        }
        if (n >= 3) {
            emitLoadStoreC(configuration, C20_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            if (k == 1) emitLoadStoreC(configuration, C20_Register, ldc, true);
            if (predicated && k == 1) backend.addInstruction(Instructions::Vector::vpst(3));
            emitLoadStoreC(configuration, C21_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            if (k == 1) emitLoadStoreC(configuration, C21_Register, ldc, true);
        }
        backend.addInstruction(Instructions::Vector::vldrw(C00_Register, C_Pointer));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        if (k == 1) emitLoadStoreC(configuration, C00_Register, ldc, true);
        // early return for k == 1
        if (k == 1) return;

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA));
        vldrImmA = 0;
        if (!aNeedsPreadd) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ldc * DT_SIZE));

        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        // handle edge case for really huge k
        if (needsDls) {
            if (kMiddle / unrollK > 65535) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (kMiddle / unrollK) >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, kMiddle / unrollK));
        }

        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n == 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]

        Instructions::Instruction16 * kLoopStart;
        if (needsDls) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < unrollK; i++) {
                if (i == 0) {
                    if (n == 1) {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                    }
                    else {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                    }
                } else {
                    if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                }
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, i * lda * DT_SIZE + 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                if (aNeedsPreadd) {
                    uint32_t skipAdds = unrollK * lda * DT_SIZE;
                    if (skipAdds <= LDR_TRESHOLD) {
                        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, skipAdds));
                    } else {
                        if (skipAdds > MOV_TRESHOLD) {
                            backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, skipAdds >> 16));
                        }
                        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, skipAdds));
                        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
                    }
                }
                if (!aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * DT_SIZE));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                if (!aNeedsPreadd && i == (unrollK - 1)) {
                    uint32_t skipAdds = unrollK * lda * DT_SIZE;
                    if (skipAdds <= LDR_TRESHOLD) {
                        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, skipAdds));
                    } else {
                        if (skipAdds > MOV_TRESHOLD) {
                            backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, skipAdds >> 16));
                        }
                        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, skipAdds));
                        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
                    }
                }
                if (aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
                bool try16 = unrollK % 2 == 0 || i < (unrollK - 1);
                if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb, false, try16); // load b[ldb]
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
                if (n == 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb, false, try16); // load b[2ldb]
                if (configuration.insertPreloadHints) backend.addInstruction(Instructions::Base::pldImmediate(A_Pointer, (i+1) * lda * DT_SIZE));
            }
        }
        if (needsDls) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);

        // process rest of k-loop
        for (uint32_t i = 0; i < kMiddle % unrollK; i++) {
            if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
            if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, i * lda * DT_SIZE + 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            // add A_Pointer, i*lda
            if (!aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * 4));
            if (n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
            if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            if (n == 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        }
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        if (predicated) { // predicate next 3 instructions
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % VECTOR_ELEMENTS));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));

            backend.addInstruction(Instructions::Vector::vpst(n >= 2 ? 3 : 2));
        }
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, aNeedsPreadd ? 4 * DT_SIZE : ((k - 2) % unrollK) * lda * DT_SIZE + (4 * DT_SIZE)));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            emitLoadStoreC(configuration, C10_Register, ldc, true);
        }
        // predicate the next rows. only needed if the rows are used (i.e. for 8x2, 8x3 microkernel)
        if (predicated && n >= 2) backend.addInstruction(Instructions::Vector::vpst(n == 3 ? 4 : 2));
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            emitLoadStoreC(configuration, C11_Register, ldc, true);
        }
        if (n == 3) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            emitLoadStoreC(configuration, C21_Register, ldc, true);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            emitLoadStoreC(configuration, C20_Register, ldc, true);
        }
    /* path for 16x1 microkernel */
    } else {
        // initialize accumulators and calculate first iteration of k

        /**
         * Used registers:
         *  - C: Q0-Q3
         *  - A: Q4-Q7
         * 
         */
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k - 2) / unrollK));

        /* First Iteration */
        for (uint32_t i = 0; i < m; i += 4) {
            Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(i / 4);
            Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(i / 4 + 4);

            backend.addInstruction(Instructions::Vector::vldrw(cReg, C_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
            if (k == 1) {
                if (predicated && i == (m - 1)) {
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % 4));
                    backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
                    backend.addInstruction(Instructions::Vector::vpst(1));
                }
                backend.addInstruction(Instructions::Vector::vstrw(cReg, C_Pointer, i * DT_SIZE));
            }
        }

        if (k == 1) return;

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));

        Instructions::Instruction16 * kLoopStart;
        if (k > 3) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            if (unrollK >= 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
            if (unrollK >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, DT_SIZE, false, true));
            if (unrollK >= 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, DT_SIZE, false, true));
            for (uint32_t i = 0; i < unrollK; i++) {
                if (i > 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                for (uint32_t j = 0; j < m; j += 4) {
                    Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(j / 4);
                    Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(j / 4 + 4);

                    backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, i * lda * DT_SIZE + j * DT_SIZE));
                    if (i == 0 || i > 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
                    if (i == 1) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B1_Register));
                    if (i == 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B2_Register));
                }
            }
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, unrollK * lda * DT_SIZE));
        }
        if (k > 3) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop
        for (uint32_t i = 0; i < kMiddle % unrollK; i++) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            for (uint32_t j = 0; j < m; j += 4) {
                Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(j / 4);
                Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(j / 4 + 4);

                backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer,  i * lda * 4 + j * 4));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
            }
        }
        // only add immediate if needed (TODO: is never needed and we can just use immediate in the next vldrw instructions)
        if (kMiddle % unrollK > 0) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, (kMiddle % unrollK) * lda * 4));

        /* Last Iteration */
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        
        if (predicated) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, m % 4));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        for (uint32_t i = predicated ? (m / 4) + 1 : m / 4; i > 0; i -= 1) {
            Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(i - 1);
            Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(i - 1 + 4);

            backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, (i-1) * 4 * 4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(cReg, C_Pointer, (i-1) * 4 * 4));
        }
    }
}

void (*JIT::Generators::Gemm::generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, bool insertPreloadHints)) (float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ c) {
    backend.resetKernel();

    // push all registers to the stack
    backend.addInstruction(JIT::Instructions::DataProcessing::push32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::LR));
    backend.addInstruction(JIT::Instructions::DataProcessing::vpush(Instructions::Q4, 4));

    MicroKernelConfiguration configuration = {};
    configuration.registerStrategy = ALL_IMMEDIATES;
    configuration.insertPreloadHints = insertPreloadHints;
    /*
    Determine where it is not possible to use immediates
    */
    /* Use 4x6 microkernel if it possible to run it every second iteration, i.e. if n minus the rest is dividible by 6 */
    bool use46Microkernel = (m <= 4 && n > 3) || // also used if m loop is not needed (or if single microkernel is generated)
        (m % DEFAULT_MICROKERNEL_M != 0 && m % DEFAULT_MICROKERNEL_M <= 4 && (n - (n % DEFAULT_MICROKERNEL_N)) % 6 == 0);
    use46Microkernel = false;
    /* We need the second B pointer if the 4x6 microkernel is used and the immediate is too large */
    bool needsBCol3Reg = use46Microkernel && 5 * DT_SIZE * k > LDR_TRESHOLD;
    /* we need the ldb register if the load from B is not possible with immediates, i.e. if k is large. is also needed whenever we have to use a second B pointer */
    bool needsLdbReg = 2 * DT_SIZE * ldb > LDR_TRESHOLD || needsBCol3Reg;
    /* if ADD can not use immediates before the VLDR */
    bool needsLdaReg = lda * DT_SIZE > LDR_TRESHOLD;
    /* Store second pointer for C */
    bool needsCRow1Reg = DT_SIZE * ldc + 16 > VLDR_TRESHOLD;
    /* Store third pointer for C */
    bool needsCRow2Reg = 2 * ldc * DT_SIZE + 16 > VLDR_TRESHOLD;
    /* CMP can only encode 255 (or an encoded immediate) */
    bool needsNReg = n > 255 && !Instructions::Base::canEncodeImmediateConstant(n);
    /* CMP can only encode 255 (or an encoded immediate) */
    bool needsMReg = m > 255 && !Instructions::Base::canEncodeImmediateConstant(m);

    /* can we unroll M and N? only unrolled if everything can be unrolled */
    bool canUnrollM = M_MAX_UNROLL * DEFAULT_MICROKERNEL_M >= m;
    bool canUnrollN = N_MAX_UNROLL * DEFAULT_MICROKERNEL_N >= n;

    uint8_t regCount = 0; // counter for used registers (max. three)
    Instructions::Register lenRegister[] = {LEN1_REGISTER, LEN2_REGISTER, LEN3_REGISTER}; // used registers

    // LDB register has highest priority
    if (needsLdbReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_LDB_REGISTER);
        configuration.LDB_REGISTER = lenRegister[regCount++];
        if (ldb > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.LDB_REGISTER, ldb >> 16));
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.LDB_REGISTER, ldb));
    }

    // BCol3 register also has high priority as it is used in k loop
    // value is calculated in the microkernel
    if (needsBCol3Reg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_BCOL3_REGISTER);
        configuration.BCOL3_REGISTER = lenRegister[regCount++];
    }

    // last important register is lda register as it is also used in the k loop
    if (needsLdaReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_A_ADD_REGISTER);
        configuration.A_ADD_REGISTER = lenRegister[regCount++];
        if (lda * DT_SIZE > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.A_ADD_REGISTER, (lda * DT_SIZE) >> 16));
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.A_ADD_REGISTER, lda * DT_SIZE));
    }

    // crow1 and crow2 pointer are important but are only needed at the start and end of the microkernel
    // value is calculated in the microkernel
    if (needsCRow1Reg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_CROW1_REGISTER);
        configuration.CROW1_REGISTER = lenRegister[regCount++];
    }

    if (needsCRow2Reg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_CROW2_REGISTER);
        configuration.CROW2_REGISTER = lenRegister[regCount++];
    }

    // not important as it only means one extra instructions per microkernel
    if (needsMReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_M_LEN_REGISTER);
        configuration.M_LEN_REGISTER = lenRegister[regCount++];
        if (m - (m % DEFAULT_MICROKERNEL_M) > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.M_LEN_REGISTER, (m - (m % DEFAULT_MICROKERNEL_M)) >> 16));
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.M_LEN_REGISTER, m - (m % DEFAULT_MICROKERNEL_M)));
    }

    // not important as it only means one extra instructions per microkernel
    if (needsNReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_N_LEN_REGISTER);
        configuration.N_LEN_REGISTER = lenRegister[regCount++];
        if (n - (n % DEFAULT_MICROKERNEL_N) > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.N_LEN_REGISTER, (n - (n % DEFAULT_MICROKERNEL_N)) >> 16));
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.N_LEN_REGISTER, n - (n % DEFAULT_MICROKERNEL_N)));
    }

    // can be solved by using a single microkernel (no loops required)
    // - 16x1
    // - 8x2, 8x3
    // - 4x4-4x6
    if ((m <= 16 && n == 1) || (m <= 8 && n <= 3) || (m <= 4 && n <= 6)) {
        generateMicroKernel(m, k, n, lda, ldb, ldc, configuration);
    } else if (n <= DEFAULT_MICROKERNEL_N) { // dont need j=n loop (only i loop) and use wide microkernel
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0)); // initialize I loop
        /*
        * Loop i (m loop): Count from 0 to m
        */
        Instructions::Instruction16 * iLoopStart;
        if (!canUnrollM) iLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32()); // start i loop
        uint32_t unrollM = canUnrollM ? (m - (m % DEFAULT_MICROKERNEL_M)) / DEFAULT_MICROKERNEL_M : 1;
        for (uint32_t i = 0; i < unrollM; i++) {
            generateMicroKernel(DEFAULT_MICROKERNEL_M, k, n, lda, ldb, ldc, configuration); // generate microkernel and pass n

            backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M)); // prepare for next iteration

            // Rewind: restore pointers or advance them
            // restore A from base pointer and calculate A[i]
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
            // Rewind B => B = B - 4*k
            if (DT_SIZE * k > LDR_TRESHOLD) {
                // if k == ldb (is the case most times), subtract via the already used register
                if (configuration.registerStrategy & USE_LDB_REGISTER && k == ldb) {
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.LDB_REGISTER, Instructions::LSL, 2));
                } else {
                    if (DT_SIZE * k > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
                }
            } else {
                backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
            }
            // to get to the next block we only have to go 8 columns forward => C += 8 * 4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));
        }
        if (!canUnrollM) {
            // ensure that only full microkernels can be executed
            uint32_t const mCmp = m - (m % DEFAULT_MICROKERNEL_M);
            // check if we need to branch back or if the loop is at its end
            if (mCmp < 255 || Instructions::Base::canEncodeImmediateConstant(mCmp)) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
            } else { // not possible to encode as immediate
                // if register can be used
                if (configuration.registerStrategy & USE_M_LEN_REGISTER) {
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, configuration.M_LEN_REGISTER));
                } else {
                    // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                    if (mCmp > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
                }
            }
            backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT); // branch back to loop start (new iteration)
        }

        // if there are remaining rows process them with an added microkernel
        if (m % DEFAULT_MICROKERNEL_M != 0) {
            generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, n, lda, ldb, ldc, configuration);
        }
    } else if (m <= DEFAULT_MICROKERNEL_M) { // dont need i=m loop (only j loop)
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(J_Loop_Register, 0)); // initialize j loop

        // select highest n for microkernel depending on the size of m
        uint32_t highestN = 0;
        if (m <= 4) highestN = n <= 6 ? n : 6;
        else highestN = n <= 3 ? n : 3;

        /*
        * Loop j (n loop): Count from 0 to n
        */
        Instructions::Instruction16 * jLoopStart;
        if (!canUnrollN) jLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32()); // start of the j loop
        uint32_t unrollN = canUnrollN ? (n - (n % DEFAULT_MICROKERNEL_N)) / DEFAULT_MICROKERNEL_N : 1;

        for (uint32_t j = 0; j < unrollN; j++) {
            generateMicroKernel(m, k, highestN, lda, ldb, ldc, configuration); // generate microkernel and pass the highestN value

            // only restore base pointer as a[i] is always a[0] because no i loop exists
            backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
            // gemm loop i end (next j)
            // Rewind B -> calculate b[j]. b is advanced in the microkernel by a whole row. so we have to move a few rows forward depending on the size of n
            uint32_t const addB = ldb * (highestN - 1) * DT_SIZE;
            if (addB > LDR_TRESHOLD) {
                if (addB > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addB >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addB));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(B_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, addB));
            }

            // C has to move forward to the next block. no m-loop so we have to add the column size
            uint32_t const addC = (highestN) * ldc * DT_SIZE;
            if (addC > LDR_TRESHOLD) {
                if (addC > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addC >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addC));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(C_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, addC));
            }
        }

        if (!canUnrollN) {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(J_Loop_Register, highestN)); // increment loop counter
            
            // ensure only full microkernels execute
            uint32_t const nCmp = n - (n % highestN);
            if (nCmp < 255 || Instructions::Base::canEncodeImmediateConstant(nCmp)) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(J_Loop_Register, nCmp));
            } else { // not possible to encode as immediate
                // if register can be used
                if (configuration.registerStrategy & USE_N_LEN_REGISTER) {
                    backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, configuration.N_LEN_REGISTER));
                } else {
                    // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                    if (nCmp > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, nCmp >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, nCmp));
                    backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, DLS_COUNT_REGISTER));
                }
            }
            backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT); // next iteration (jump back to beginning of the loop)
        }

        // handle j loop edge cases
        if (n % highestN != 0) {
            generateMicroKernel(m, k, n % highestN, lda, ldb, ldc, configuration);
        } 
    } else { // both i and j loop needed (normally the case if both m and n are large enough)
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(J_Loop_Register, 0)); // initialize j loop counter

        /*
        * Loop j (n loop): Count from 0 to n
        */
        Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0)); // start j loop: initialize i loop counter
        uint32_t unrollN = canUnrollN ? (n - (n % DEFAULT_MICROKERNEL_N)) / DEFAULT_MICROKERNEL_N : 1;
        for (uint32_t j = 0; j < unrollN; j++) {
            if (j > 0) backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0)); // initialize i loop counter in each iteration
            /*
            * Loop i (m loop): Count from 0 to m
            */
            Instructions::Instruction16 * iLoopStart;
            if (!canUnrollM) iLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32()); // start i loop
            uint32_t unrollM = canUnrollM ? (m - (m % DEFAULT_MICROKERNEL_M)) / DEFAULT_MICROKERNEL_M : 1;
            for (uint32_t i = 0; i < unrollM; i++) {
                generateMicroKernel(DEFAULT_MICROKERNEL_M, k, DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration); // generate microkernel with default parameters

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M)); // increment i loop counter
                
                // Rewind
                // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
                backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
                // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
                // Rewind B => B = B - 4*k
                // b will be incremented in each k-Iteration by ldr r7, [r1], #4. this is done k times and needs to be reverted
                if (DT_SIZE * k > LDR_TRESHOLD) {
                    if (configuration.registerStrategy & USE_LDB_REGISTER && k == ldb) {
                        backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.LDB_REGISTER, Instructions::LSL, 2));
                    } else {
                        if (DT_SIZE * k > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                        backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
                    }
                } else {
                    backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
                }
                // Forward C => C += 8*4
                // After each block is computed in the m-loop, C has to move forward to the next block
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));
            }

            // ensure that only full microkernels can be executed
            if (!canUnrollM) {
                uint32_t const mCmp = m - (m % DEFAULT_MICROKERNEL_M);
                if (mCmp < 255) {
                    backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
                } else if (Instructions::Base::canEncodeImmediateConstant(mCmp)) {
                    backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
                } else { // not possible to encode as immediate
                    // if register can be used
                    if (configuration.registerStrategy & USE_M_LEN_REGISTER) {
                        backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, configuration.M_LEN_REGISTER));
                    } else {
                        // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                        if (mCmp > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                        backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
                    }
                }
                backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT); // next iteration of i loop
            }

            // handle i loop edge cases
            if (m % DEFAULT_MICROKERNEL_M != 0) {
                Instructions::Instruction16 * branchToTailEnd;
                // if we use 4x6 microkernel check if the kernel shall be executed in the current iteration or we have to skip in this iteration
                // this is done by AND and checking if the loop counter is even. if it is even (0, 6, 12, ...) we can execute the 4x6 microkernel
                // if it is uneven (3, 9, 15, ...) we have to skip
                // if unrolling is used, the branching logic can be skipped and we insert the microkernel at each position
                if (use46Microkernel && !canUnrollN) {
                    backend.addInstruction(Instructions::Arithmetic::andImmediate32(DLS_COUNT_REGISTER, J_Loop_Register, 1));
                    backend.addInstruction(Instructions::Base::cmpImmediate32(DLS_COUNT_REGISTER, 1));
                    // insert placeholder for forward branch. if we dont execute in this iteration we have to skip to the next iteration (handling for this is at the end)
                    branchToTailEnd = backend.addBranchPlaceholder(); // beq tailEnd
                }
                // generate edge case microkernel and use 4x6 if possible
                // if unrolled only insert in correct places
                if (!use46Microkernel || !canUnrollN || (canUnrollN && use46Microkernel && j % 2 == 0)) {
                    generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, use46Microkernel ? 6 : DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);

                    // Rewind
                    // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
                    // Rewind B => B = B - 4*k
                    if (DT_SIZE * k > LDR_TRESHOLD) {
                        if (configuration.registerStrategy & USE_LDB_REGISTER && k == ldb) {
                            backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.LDB_REGISTER, Instructions::LSL, 2));
                        } else {
                            if (DT_SIZE * k > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                            backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
                        }
                    } else {
                        backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
                    }
                }
                // Rewind C => C += 8*4
                // jump point if 4x6 microkernel is used and not executed in current iteration
                Instructions::Instruction16 * tailEnd46 = backend.addBranchTargetInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (m % DEFAULT_MICROKERNEL_M) * DT_SIZE));
                if (use46Microkernel && !canUnrollN) backend.setForwardsBranch(branchToTailEnd, tailEnd46, Instructions::EQ);
            }

            // gemm loop i end (next j)
            // Rewind A -> already rewinded by i so need to reset
            backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
            // Rewind B -> rewinded by i to start. add 3*len
            uint32_t const addB = ldb * DEFAULT_MICROKERNEL_N * DT_SIZE;
            if (addB > LDR_TRESHOLD) {
                if (addB > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addB >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addB));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(B_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, addB));
            }
            // Rewind C -> still have to go two lines -> 2ldc
            uint32_t const addC = (DEFAULT_MICROKERNEL_N - 1) * ldc * DT_SIZE;
            if (addC > LDR_TRESHOLD) {
                if (addC > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addC >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addC));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(C_Pointer, DLS_COUNT_REGISTER));
            } else {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, addC));
            }
            // increment j loop counter (needed for 4x6 microkernel even if unrolled)
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(J_Loop_Register, DEFAULT_MICROKERNEL_N)); 
        }

        if (!canUnrollN) {
            // ensure only full microkernels are executed
            uint32_t const nCmp = n - (n % DEFAULT_MICROKERNEL_N);
            if (nCmp < 255) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(J_Loop_Register, nCmp));
            } else if (Instructions::Base::canEncodeImmediateConstant(nCmp)) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(J_Loop_Register, nCmp));
            } else { // not possible to encode as immediate
                // if register can be used
                if (configuration.registerStrategy & USE_N_LEN_REGISTER) {
                    backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, configuration.N_LEN_REGISTER));
                } else {
                    // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                    if (nCmp > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, nCmp >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, nCmp));
                    backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, DLS_COUNT_REGISTER));
                }
            }
            backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);
        }

        // handle j loop edge cases
        if (n % DEFAULT_MICROKERNEL_N != 0) {
            // new i loop is created for 8x2/8x1 microkernel over which is looped 
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0)); //  loop counter for new i loop
            Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Base::nop32());

            generateMicroKernel(DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
            
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M));

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));

            // Rewind B => B = B - 4*k
            // not possible to subtract immediate
            if (DT_SIZE * k > LDR_TRESHOLD) {
                if (configuration.registerStrategy & USE_LDB_REGISTER && k == ldb) { // can utilize k register
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.LDB_REGISTER, Instructions::LSL, 2));
                } else {
                    // TODO provide slow fallback (will never happen as K Register has highest priority)
                }
            } else {
                backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
            }

            // Rewind C => C += 8*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));

            // ensure that only full microkernels can be executed
            uint32_t const mCmp = m - (m % DEFAULT_MICROKERNEL_M);
            if (mCmp < 255) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
            } else if (Instructions::Base::canEncodeImmediateConstant(mCmp)) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
            } else { // not possible to encode as immediate
                // if register can be used
                if (configuration.registerStrategy & USE_M_LEN_REGISTER) {
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, configuration.M_LEN_REGISTER));
                } else {
                    // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                    if (mCmp > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
                }
            }

            backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

            // last corner
            if (m % DEFAULT_MICROKERNEL_M != 0) {
                generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
            }
        }
    }

    // gemm loop j end
    backend.addInstruction(Instructions::DataProcessing::vpop(Instructions::Q4, 4));
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::PC));

    backend.clearCaches();
    return reinterpret_cast<Func>(backend.getThumbAddress());
}
