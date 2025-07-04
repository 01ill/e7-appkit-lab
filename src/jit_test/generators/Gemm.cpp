#include "Gemm.hpp"
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
constexpr JIT::Instructions::Register B0_Register = JIT::Instructions::R7;
constexpr JIT::Instructions::Register B1_Register = JIT::Instructions::R8;
constexpr JIT::Instructions::Register B2_Register = JIT::Instructions::R9;
constexpr JIT::Instructions::Register I_Loop_Register = JIT::Instructions::R5;
constexpr JIT::Instructions::Register J_Loop_Register = JIT::Instructions::R4;
constexpr JIT::Instructions::Register A_Pointer = JIT::Instructions::R0;
constexpr JIT::Instructions::Register B_Pointer = JIT::Instructions::R1;
constexpr JIT::Instructions::Register C_Pointer = JIT::Instructions::R2;
constexpr JIT::Instructions::Register DLS_COUNT_REGISTER = JIT::Instructions::R6;
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
/* Limit K unrolling to limit the buffer size */
constexpr uint32_t K_MAX_UNROLL = 5;
/* Use 8x3 microkernel by default */
constexpr uint32_t DEFAULT_MICROKERNEL_M = 8;
constexpr uint32_t DEFAULT_MICROKERNEL_N = 3;

constexpr uint32_t VECTOR_SIZE = 16; // == 128 Bit
/* Count of vector registers */
constexpr uint32_t VECTOR_COUNT = 8;
constexpr uint32_t DT_SIZE = 4; // == 32 Bit (FP32)
/* Calculate elements of elements which fit into a single vector register */
constexpr uint32_t VECTOR_ELEMENTS = VECTOR_SIZE / DT_SIZE;

void JIT::Generators::Gemm::emitLoadB(JIT::Instructions::Register targetReg, MicroKernelConfiguration & configuration, uint32_t leftShiftAmount, uint32_t offset, bool secondHalf) {
    if ((configuration.registerStrategy & (secondHalf ? USE_K_LEN2_REGISTER : USE_K_LEN_REGISTER)) && (targetReg == B1_Register || targetReg == B2_Register)) {
        backend.addInstruction(Instructions::DataProcessing::ldrRegister32(targetReg, secondHalf ? configuration.K_LEN2_REGISTER : B_Pointer, configuration.K_LEN_REGISTER, leftShiftAmount));
    } else if (targetReg == B1_Register || targetReg == B2_Register) {
        if (offset <= LDR_TRESHOLD) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer , offset));
        } else {
            // TODO: slow fallback (will never happen anyways has K has priority)
            Instructions::Base::printValidationError("emitLoadB Immediate Fallback; inserting nop");
            backend.addInstruction(Instructions::Base::nop32());
        }
    } else { // if targetReg == B0_Register
        if (configuration.registerStrategy & USE_K_LEN_REGISTER && !secondHalf) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer, DT_SIZE, false, true));
        } else if (configuration.registerStrategy & USE_K_LEN2_REGISTER && secondHalf) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, configuration.K_LEN2_REGISTER, DT_SIZE, false, true));
        } else if (secondHalf) {
            if (offset <= LDR_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(targetReg, B_Pointer, offset));
            } else {
                // TODO: slow fallback (will never happen anyways has K has priority)
                Instructions::Base::printValidationError("emitLoadB Immediate Fallback; inserting nop");
                backend.addInstruction(Instructions::Base::nop32());
            }
        }
    }
}

// void JIT::Generators::Gemm::emitLoadB46() {

// }

void JIT::Generators::Gemm::emitLoadStoreC(MicroKernelConfiguration & configuration, Instructions::VectorRegister targetReg, uint32_t ldc, bool store) {
    #ifdef FLAG_NO_C_LOAD
    if (!store) {
        backend.addInstruction(Instructions::Vector::vmovImmediate(targetReg, 0, Instructions::I32));
        return;
    }
    #endif
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
            // TODO: slow fallback (will never happen anyways as CROW has priority)
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(DLS_COUNT_REGISTER, C_Pointer, offset));
            if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, DLS_COUNT_REGISTER));
            else backend.addInstruction(Instructions::Vector::vldrw(targetReg, DLS_COUNT_REGISTER));
        }
    }
}

void JIT::Generators::Gemm::emitLoadStoreC46(Instructions::VectorRegister targetReg, uint32_t ldc, bool store) {
    uint32_t imm = ldc * DT_SIZE;
    if (imm > VLDR_TRESHOLD) {
        if (store) backend.addInstruction(Instructions::Vector::vstrw(targetReg, C_Pointer));
        else backend.addInstruction(Instructions::Vector::vldrw(targetReg, C_Pointer));
        if (imm > LDR_TRESHOLD) {
            // backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, imm));
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
    uint32_t neededVectorRegisters = m % VECTOR_ELEMENTS != 0 
        ? ((m / VECTOR_ELEMENTS) + 1) * n + ((m / VECTOR_ELEMENTS) + 1) // with predicates
        : (m / VECTOR_ELEMENTS) * n + (m / VECTOR_ELEMENTS); // no predicates. all vector registers are fully used
    bool aNeedsPreadd = lda * DT_SIZE > VLDR_TRESHOLD;
    // if not all elements fit into a single vector register, the instructions have to be predicated
    // the helium vector register can hold 4 fp32 values
    bool predicated = m % VECTOR_ELEMENTS != 0;
    uint32_t maxSkippedAdds = VLDR_TRESHOLD / (DT_SIZE * lda);
    maxSkippedAdds = maxSkippedAdds > K_MAX_UNROLL ? K_MAX_UNROLL : maxSkippedAdds; // limit k unrolling (priorize code size over (really small) performance gain)
    maxSkippedAdds = maxSkippedAdds > (k-2) ? (k-2) : maxSkippedAdds; // limit unrolling if k is small
    maxSkippedAdds = maxSkippedAdds < 1 ? 1 : maxSkippedAdds; // at least one iteration
    bool needsDls = (k - 2) / maxSkippedAdds > 1;
    if (neededVectorRegisters > VECTOR_COUNT) {
        Instructions::Base::printValidationError("generateMicroKernel: dimensions too high - cant fit in vector registers");
        backend.addInstruction(Instructions::Base::nop32());
        return;
    }

    /* Path for 4x4-4x6 microkernel */
    if (m <= 4) {
        if (configuration.registerStrategy & USE_K_LEN2_REGISTER) {
            uint32_t imm = 3 * ldb * DT_SIZE;
            if (imm > LDR_TRESHOLD) {
                if (imm > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.K_LEN2_REGISTER, imm >> 16));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.K_LEN2_REGISTER, imm));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(configuration.K_LEN2_REGISTER, B_Pointer));
            } else backend.addInstruction(Instructions::Arithmetic::addImmediate32(configuration.K_LEN2_REGISTER, B_Pointer, imm));
        }

        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
        backend.addHeliumInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        uint32_t vldrImmA = ldc * DT_SIZE;
        if (configuration.registerStrategy & USE_A_ADD_REGISTER) {
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, configuration.A_ADD_REGISTER));
            vldrImmA = 0;
        } else if (aNeedsPreadd) {
            if (vldrImmA < LDR_TRESHOLD) {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * DT_SIZE));
                vldrImmA = 0;
            } else {
                // TODO: slow fallback (will never happen anyways as A has priority)
                Instructions::Base::printValidationError("loadA Immediate Fallback; inserting nop");
                backend.addInstruction(Instructions::Base::nop32());
            }
            vldrImmA = 0;
        }
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        emitLoadStoreC46(C00_Register, ldc);
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        if (n >= 2) {
            emitLoadStoreC46(C10_Register, ldc);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
        }
        if (n >= 3) {
            emitLoadStoreC46(C20_Register, ldc);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
        }

        if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true);
        if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true);
        if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true);
        
        if (n >= 4) {
            emitLoadStoreC46(C30_Register, ldc);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register));
        }
        if (n >= 5) {
            emitLoadStoreC46(C40_Register, ldc);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register));
        }
        if (n >= 6) {
            emitLoadStoreC46(C50_Register, ldc);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register));
        }

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA));
        vldrImmA = 0;
        if (!aNeedsPreadd) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * DT_SIZE));

        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        // handle edge case for really huge k
        if (needsDls) {
            if ((k-2) / maxSkippedAdds > 65535) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, ((k-2) / maxSkippedAdds) >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k-2) / maxSkippedAdds));
        }

        /* Load B */
        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]

        Instructions::Instruction16 * kLoopStart;
        if (needsDls) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
                if (i == 0) {
                    if (n == 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                    else {
                        kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                    }
                } else {
                    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
                }
                if (n >= 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                if (n >= 2 && n < 4) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb] (different order for 4x3)
                if (n >= 5) emitLoadB(B1_Register, configuration, 2, 4 * ldb * DT_SIZE - 4, true);
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                if (n >= 6) emitLoadB(B2_Register, configuration, 3, 5 * ldb * DT_SIZE - 4, true);
                if (n >= 5) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C40_Register, A0_Register, B1_Register));
                if (n >= 4) emitLoadB(B0_Register, configuration, 0, 3 * ldb * DT_SIZE - 4, true);
                if (n >= 6) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C50_Register, A0_Register, B2_Register));
                if (n >= 4) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb] (different order for 4x6)
                if (n >= 4) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C30_Register, A0_Register, B0_Register));
                if (!aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, (i+1) * lda * DT_SIZE));
                if (i < (maxSkippedAdds-1) && n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                if (n >= 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
            }
            uint32_t skipAdds = maxSkippedAdds * lda * DT_SIZE;
            if (skipAdds <= LDR_TRESHOLD) {
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, skipAdds));
            } else {
                if (skipAdds > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, skipAdds >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, skipAdds));
                backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, DLS_COUNT_REGISTER));
            }
            if (aNeedsPreadd) backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        }
        if (needsDls) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);
        
        // process rest of k-loop
        for (uint32_t i = 0; i < (k-2) % maxSkippedAdds; i++) {
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
            if (i < (((k-2) % maxSkippedAdds)-1) && n >= 2) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
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
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ldc * DT_SIZE));
                vldrImmA = 0;
            } else {
                // TODO: slow fallback (will never happen anyways as A has priority)
                Instructions::Base::printValidationError("Load Immediate Fallback; inserting nop");
                backend.addInstruction(Instructions::Base::nop32());
            }
            vldrImmA = 0;
        }
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
        backend.addInstruction(Instructions::Vector::vldrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        if (n >= 2) {
            emitLoadStoreC(configuration, C10_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));

            emitLoadStoreC(configuration, C11_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
        }
        if (n >= 3) {
            emitLoadStoreC(configuration, C20_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            emitLoadStoreC(configuration, C21_Register, ldc, false);
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
        }
        backend.addInstruction(Instructions::Vector::vldrw(C00_Register, C_Pointer));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));

        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, vldrImmA));
        vldrImmA = 0;
        if (!aNeedsPreadd) backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, ldc * DT_SIZE));

        // if the next vldrw can be used with an immediate, use the immediate instead of the add instruction
        // to use this, we need to unroll the k loop
        // handle edge case for really huge k
        if (needsDls) {
            if ((k-2) / maxSkippedAdds > 65535) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, ((k-2) / maxSkippedAdds) >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k-2) / maxSkippedAdds));
        }

        if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
        if (n == 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]

        Instructions::Instruction16 * kLoopStart;
        if (needsDls) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
                if (i == 0) {
                    if (n == 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
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
                    uint32_t skipAdds = maxSkippedAdds * lda * DT_SIZE;
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
                if (!aNeedsPreadd && i == (maxSkippedAdds - 1)) {
                    uint32_t skipAdds = maxSkippedAdds * lda * DT_SIZE;
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
                if (n >= 2) emitLoadB(B1_Register, configuration, 2, DT_SIZE * ldb); // load b[ldb]
                if (n == 3) backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
                if (n == 3) emitLoadB(B2_Register, configuration, 3, 2 * DT_SIZE * ldb); // load b[2ldb]
            }
        }
        if (needsDls) backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);

        // process rest of k-loop
        for (uint32_t i = 0; i < (k-2) % maxSkippedAdds; i++) {
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
            backend.addInstruction(Instructions::Vector::vpst(3));
        }
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, aNeedsPreadd ? 4 * DT_SIZE : ((k - 2) % maxSkippedAdds) * lda * DT_SIZE + (4 * DT_SIZE)));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
        if (n >= 2) {
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            emitLoadStoreC(configuration, C10_Register, ldc, true);
        }
        if (predicated && n >= 2) { // predicate the next rows. only needed if the rows are used (i.e. for 8x2, 8x3 microkernel)
            // backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, DLS_COUNT_REGISTER));
            backend.addInstruction(Instructions::Vector::vpst(n == 3 ? 4 : 2));
        }
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
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, (k - 2) / maxSkippedAdds));

        /* First Iteration */
        for (uint32_t i = 0; i < m; i += 4) {
            Instructions::VectorRegister cReg = static_cast<Instructions::VectorRegister>(i / 4);
            Instructions::VectorRegister aReg = static_cast<Instructions::VectorRegister>(i / 4 + 4);

            backend.addInstruction(Instructions::Vector::vldrw(cReg, C_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vldrw(aReg, A_Pointer, i * DT_SIZE));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(cReg, aReg, B0_Register));
        }

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda * 4));

        Instructions::Instruction16 * kLoopStart;
        if (k > 3) backend.addInstruction(Instructions::Base::dls(DLS_COUNT_REGISTER));
        if (k >= 3) {
            if (maxSkippedAdds >= 1) kLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, DT_SIZE, false, true));
            if (maxSkippedAdds >= 2) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, DT_SIZE, false, true));
            if (maxSkippedAdds >= 3) backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, DT_SIZE, false, true));
            for (uint32_t i = 0; i < maxSkippedAdds; i++) {
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

void (*JIT::Generators::Gemm::generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc)) (float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ c) {
    backend.resetKernel();

    /*
     * INIT:
     * - Push callee saved registers and vector registers
     * - save base pointers
     * - initialize j counter
     */
    backend.addInstruction(JIT::Instructions::DataProcessing::push32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::LR));
    backend.addInstruction(JIT::Instructions::DataProcessing::vpush(Instructions::Q4, 4));

    MicroKernelConfiguration configuration = {};
    configuration.registerStrategy = ALL_IMMEDIATES;
    /*
    Select Strategy
    */
    bool use46Microkernel = m % DEFAULT_MICROKERNEL_M != 0 && m % DEFAULT_MICROKERNEL_M <= 4 && (n - (n % DEFAULT_MICROKERNEL_N)) % 6 == 0;
    bool needsKReg2 = use46Microkernel && 5 * DT_SIZE * k > LDR_TRESHOLD;
    bool needsKReg = 2 * DT_SIZE * ldb > LDR_TRESHOLD || needsKReg2; // K_LEN2 still needs the value of ldb stored.
    bool needsAReg = lda * DT_SIZE > LDR_TRESHOLD;
    bool needsCRow1Reg = DT_SIZE * ldc + 16 > VLDR_TRESHOLD;
    bool needsCRow2Reg = 2 * ldc * DT_SIZE + 16 > VLDR_TRESHOLD;
    bool needsNReg = n > 255 && !Instructions::Base::canEncodeImmediateConstant(n);
    bool needsMReg = m > 255 && !Instructions::Base::canEncodeImmediateConstant(m);

    uint8_t regCount = 0;
    Instructions::Register lenRegister[] = {LEN1_REGISTER, LEN2_REGISTER, LEN3_REGISTER};

    if (needsKReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_K_LEN_REGISTER);
        configuration.K_LEN_REGISTER = lenRegister[regCount++];
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.K_LEN_REGISTER, ldb));
    }

    if (needsKReg2 && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_K_LEN2_REGISTER);
        configuration.K_LEN2_REGISTER = lenRegister[regCount++];
        uint32_t imm = 3 * ldb * DT_SIZE;
        if (imm > LDR_TRESHOLD) {
            if (imm > MOV_TRESHOLD) backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.K_LEN2_REGISTER, imm >> 16));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.K_LEN2_REGISTER, imm));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(configuration.K_LEN2_REGISTER, B_Pointer));
        } else backend.addInstruction(Instructions::Arithmetic::addImmediate32(configuration.K_LEN2_REGISTER, B_Pointer, imm));
    }

    if (needsAReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_A_ADD_REGISTER);
        configuration.A_ADD_REGISTER = lenRegister[regCount++];
        if (lda * DT_SIZE > MOV_TRESHOLD) {
            backend.addInstruction(Instructions::DataProcessing::movtImmediate32(configuration.A_ADD_REGISTER, (lda * DT_SIZE) >> 16));
        }
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.A_ADD_REGISTER, lda * DT_SIZE));
    }

    if (needsCRow1Reg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_CROW1_REGISTER);
        configuration.CROW1_REGISTER = lenRegister[regCount++];
    }

    if (needsCRow2Reg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_CROW2_REGISTER);
        configuration.CROW2_REGISTER = lenRegister[regCount++];
    }

    if (needsMReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_M_LEN_REGISTER);
        configuration.M_LEN_REGISTER = lenRegister[regCount++];
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.M_LEN_REGISTER, m - (m % DEFAULT_MICROKERNEL_M)));
    }

    if (needsNReg && regCount < 3) {
        configuration.registerStrategy = static_cast<RegisterImmediateStrategy>(configuration.registerStrategy | USE_N_LEN_REGISTER);
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(configuration.N_LEN_REGISTER, n - (n % DEFAULT_MICROKERNEL_N)));
        configuration.N_LEN_REGISTER = lenRegister[regCount++];
    }

    // can be solved by using a single microkernel (no loops required)
    // - 16x1
    // - 8x2, 8x3
    // - 4x4-4x6
    if ((m <= 16 && n == 1) || (m <= 8 && n <= 3) || (m <= 4 && n <= 6)) {
        generateMicroKernel(m, k, n, lda, ldb, ldc, configuration);
    } else if (n <= DEFAULT_MICROKERNEL_N) { // dont need j=n loop (only i loop) and use wide microkernel
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));
        Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Base::nop32());

        generateMicroKernel(DEFAULT_MICROKERNEL_M, k, n, lda, ldb, ldc, configuration);

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, DEFAULT_MICROKERNEL_M));

        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        if (DT_SIZE * k > LDR_TRESHOLD) {
            if (configuration.registerStrategy & USE_K_LEN_REGISTER && k == ldb) {
                backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.K_LEN_REGISTER, Instructions::LSL, 2));
            } else {
                if (DT_SIZE * k > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
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
                if (mCmp > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
            }
        }
        backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

        if (m % DEFAULT_MICROKERNEL_M != 0) {
            generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, n, lda, ldb, ldc, configuration);
        }
    } else if (m <= DEFAULT_MICROKERNEL_M) { // dont need i=m loop (only j loop)
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(J_Loop_Register, 0)); // mov 0 to r4

        // select highest n
        uint32_t highestN = 0;
        if (m <= 4) {
            highestN = n <= 6 ? n : 6;
        } else {
            highestN = n <= 3 ? n : 3;
        }

        /*
        * Loop j (n loop): Count from 0 to n
        */
        Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32());

        generateMicroKernel(m, k, highestN, lda, ldb, ldc, configuration);

        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        // b will be incremented in each k-Iteration by ldr r7, [r1], #4. this is done k times and needs to be reverted

        // gemm loop i end (next j)
        // Rewind A -> already rewinded by i so need to reset
        // Rewind B -> rewinded by i to start. add 3*len
        uint32_t const addB = ldb * (highestN-1) * DT_SIZE;
        if (addB > LDR_TRESHOLD) {
            if (addB > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addB >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addB));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(B_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, addB));
        }
        // Forward C => C += 8*4
        // After each block is computed in the m-loop, C has to move forward to the next block
        // Rewind C -> still have to go two lines -> 2ldc
        uint32_t const addC = (highestN) * ldc * DT_SIZE;
        if (addC > LDR_TRESHOLD) {
            if (addC > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addC >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addC));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(C_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, addC));
        }

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(J_Loop_Register, highestN));
        
        uint32_t const nCmp = n - (n % highestN);
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
                if (nCmp > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, nCmp >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, nCmp));
                backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, DLS_COUNT_REGISTER));
            }
        }

        backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);

        // handle j loop edge cases
        if (n % highestN != 0) {
            generateMicroKernel(m, k, n % highestN, lda, ldb, ldc, configuration);
        } 
    } else {
        backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
        backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate16(J_Loop_Register, 0)); // mov 0 to r4

        /*
        * Loop j (n loop): Count from 0 to n
        */
        Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::movImmediate16(I_Loop_Register, 0));

        /*
        * Loop i (m loop): Count from 0 to m
        */
        Instructions::Instruction16 * iLoopStart = backend.addBranchTargetInstruction(Instructions::Base::nop32());

        generateMicroKernel(DEFAULT_MICROKERNEL_M, k, DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
        backend.addInstruction(Instructions::Arithmetic::addImmediate16(I_Loop_Register, DEFAULT_MICROKERNEL_M));
        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        // b will be incremented in each k-Iteration by ldr r7, [r1], #4. this is done k times and needs to be reverted
        if (DT_SIZE * k > LDR_TRESHOLD) {
            if (configuration.registerStrategy & USE_K_LEN_REGISTER && k == ldb) {
                backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.K_LEN_REGISTER, Instructions::LSL, 2));
            } else {
                if (DT_SIZE * k > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
            }
        } else {
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
        }
        // Forward C => C += 8*4
        // After each block is computed in the m-loop, C has to move forward to the next block
        backend.addInstruction(Instructions::Arithmetic::addImmediate16(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));

        // ensure that only full microkernels can be executed
        uint32_t const mCmp = m - (m % DEFAULT_MICROKERNEL_M);
        if (mCmp < 255) {
            backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, mCmp));
        } else if (Instructions::Base::canEncodeImmediateConstant(mCmp)) {
            backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
        } else { // not possible to encode as immediate
            // if register can be used
            if (configuration.registerStrategy & USE_M_LEN_REGISTER) {
                backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, configuration.M_LEN_REGISTER));
            } else {
                // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                if (mCmp > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
            }
        }
        backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT);

        // handle i loop edge cases
        if (m % DEFAULT_MICROKERNEL_M != 0) {
            // Check if we can use 4x6 microkernel to boost performance
            Instructions::Instruction16 * branchToTailEnd;
            if (use46Microkernel) {
                backend.addInstruction(Instructions::Arithmetic::andImmediate32(DLS_COUNT_REGISTER, J_Loop_Register, 1));
                backend.addInstruction(Instructions::Base::cmpImmediate32(DLS_COUNT_REGISTER, 1));
                branchToTailEnd = backend.addBranchPlaceholder(); // afterwards we insert beq tail_end

                generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, 6, lda, ldb, ldc, configuration);
            } else {
                generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
            }

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            // Rewind B => B = B - 4*k
            if (DT_SIZE * k > LDR_TRESHOLD) {
                if (configuration.registerStrategy & USE_K_LEN_REGISTER && k == ldb) {
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.K_LEN_REGISTER, Instructions::LSL, 2));
                } else {
                    if (DT_SIZE * k > MOV_TRESHOLD) {
                        backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, (DT_SIZE * k) >> 16));
                    }
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, DT_SIZE * k));
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, DLS_COUNT_REGISTER));
                }
            } else {
                backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
            }
            // Rewind C => C += 8*4
            // jump point if 4x6 microkernel is used and not executed in current iteration
            Instructions::Instruction16 * tailEnd46 = backend.addBranchTargetInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (m % DEFAULT_MICROKERNEL_M) * DT_SIZE));
            if (use46Microkernel) {
                backend.setForwardsBranch(branchToTailEnd, tailEnd46, Instructions::EQ);
            }
        }

        // gemm loop i end (next j)
        // Rewind A -> already rewinded by i so need to reset
        backend.addInstruction(Instructions::DataProcessing::movRegister16(A_Pointer, A_Base_Pointer));
        // Rewind B -> rewinded by i to start. add 3*len
        uint32_t const addB = ldb * DEFAULT_MICROKERNEL_N * DT_SIZE;
        if (addB > LDR_TRESHOLD) {
            if (addB > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addB >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addB));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(B_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, addB));
        }
        // Rewind C -> still have to go two lines -> 2ldc
        uint32_t const addC = (DEFAULT_MICROKERNEL_N - 1) * ldc * DT_SIZE;
        if (addC > LDR_TRESHOLD) {
            if (addC > MOV_TRESHOLD) {
                backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, addC >> 16));
            }
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, addC));
            backend.addInstruction(Instructions::Arithmetic::addRegister32(C_Pointer, DLS_COUNT_REGISTER));
        } else {
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, addC));
        }

        backend.addInstruction(Instructions::Arithmetic::addImmediate16(J_Loop_Register, DEFAULT_MICROKERNEL_N));
        
        uint32_t const nCmp = n - (n % DEFAULT_MICROKERNEL_N);
        if (nCmp < 255) {
            backend.addInstruction(Instructions::Base::cmpImmediate16(J_Loop_Register, nCmp));
        } else if (Instructions::Base::canEncodeImmediateConstant(nCmp)) {
            backend.addInstruction(Instructions::Base::cmpImmediate32(J_Loop_Register, nCmp));
        } else { // not possible to encode as immediate
            // if register can be used
            if (configuration.registerStrategy & USE_N_LEN_REGISTER) {
                backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, configuration.N_LEN_REGISTER));
            } else {
                // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                if (nCmp > MOV_TRESHOLD) {
                    backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, nCmp >> 16));
                }
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, nCmp));
                backend.addInstruction(Instructions::Base::cmpRegister32(J_Loop_Register, DLS_COUNT_REGISTER));
            }
        }

        backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);

        // handle j loop edge cases
        if (n % DEFAULT_MICROKERNEL_N != 0) {
            backend.addInstruction(Instructions::DataProcessing::movImmediate16(I_Loop_Register, 0));
            Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Base::nop16());

            generateMicroKernel(DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
            
            backend.addInstruction(Instructions::Arithmetic::addImmediate16(I_Loop_Register, DEFAULT_MICROKERNEL_M));

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
            // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
            // Rewind B => B = B - 4*k
            // not possible to subtract immediate
            if (DT_SIZE * k > LDR_TRESHOLD) {
                if (configuration.registerStrategy & USE_K_LEN_REGISTER && k == ldb) { // can utilize k register
                    backend.addInstruction(Instructions::Arithmetic::subRegister32(B_Pointer, configuration.K_LEN_REGISTER, Instructions::LSL, 2));
                } else {
                    // TODO provide slow fallback (will never happen as K Register has highest priority)
                }
            } else {
                backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, DT_SIZE * k));
            }

            // Rewind C => C += 8*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate16(C_Pointer, DEFAULT_MICROKERNEL_M * DT_SIZE));

            // ensure that only full microkernels can be executed

            uint32_t const mCmp = m - (m % DEFAULT_MICROKERNEL_M);
            if (mCmp < 255) {
                backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, mCmp));
            } else if (Instructions::Base::canEncodeImmediateConstant(mCmp)) {
                backend.addInstruction(Instructions::Base::cmpImmediate32(I_Loop_Register, mCmp));
            } else { // not possible to encode as immediate
                // if register can be used
                if (configuration.registerStrategy & USE_M_LEN_REGISTER) {
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, configuration.M_LEN_REGISTER));
                } else {
                    // if no register can be used then just repurpose the DLS_COUNT_REGISTER as it will be written again in the next iteration anyways
                    if (mCmp > MOV_TRESHOLD) {
                        backend.addInstruction(Instructions::DataProcessing::movtImmediate32(DLS_COUNT_REGISTER, mCmp >> 16));
                    }
                    backend.addInstruction(Instructions::DataProcessing::movImmediate32(DLS_COUNT_REGISTER, mCmp));
                    backend.addInstruction(Instructions::Base::cmpRegister32(I_Loop_Register, DLS_COUNT_REGISTER));
                }
            }

            backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

            if (m % DEFAULT_MICROKERNEL_M != 0) {
                generateMicroKernel(m % DEFAULT_MICROKERNEL_M, k, n % DEFAULT_MICROKERNEL_N, lda, ldb, ldc, configuration);
            }
        }
    }

    // gemm loop j end
    backend.addInstruction(Instructions::DataProcessing::vpop(Instructions::Q4, 4));
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::PC));

    __asm("dsb");
    __asm("isb");
    return reinterpret_cast<Func>(backend.getThumbAddress());
}
