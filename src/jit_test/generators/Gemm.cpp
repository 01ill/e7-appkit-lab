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
constexpr JIT::Instructions::Register B1_Register = JIT::Instructions::R7;
constexpr JIT::Instructions::Register B2_Register = JIT::Instructions::R8;
constexpr JIT::Instructions::Register B0_Register = JIT::Instructions::R9;
constexpr JIT::Instructions::Register I_Loop_Register = JIT::Instructions::R5;
constexpr JIT::Instructions::Register J_Loop_Register = JIT::Instructions::R4;
constexpr JIT::Instructions::Register A_Pointer = JIT::Instructions::R0;
constexpr JIT::Instructions::Register B_Pointer = JIT::Instructions::R1;
constexpr JIT::Instructions::Register C_Pointer = JIT::Instructions::R2;
constexpr JIT::Instructions::Register A_Base_Pointer = JIT::Instructions::R10;


void (*JIT::Generators::Gemm::generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc)) (float const * a, float const * b, float * c) {
    backend.resetKernel();

    /*
     * INIT:
     * - Push callee saved registers and vector registers
     * - save base pointers
     * - initialize j counter
     */
    backend.addInstruction(JIT::Instructions::DataProcessing::push32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::LR));
    backend.addInstruction(JIT::Instructions::DataProcessing::vpush(Instructions::Q4, 4));
    backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(A_Base_Pointer, A_Pointer)); // save a pointer
    backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(J_Loop_Register, 0)); // mov 0 to r4
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R6, k-2));
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R3, ldb));

    /*
    * Loop j (n loop): Count from 0 to n
    */
    Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));

    /*
    * Loop i (m loop): Count from 0 to m
    */
    Instructions::Instruction16 * iLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
    // load b[2len]
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 8*ldb));
    // load b[0] and write back
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

    // init accumulators
    backend.addInstruction(Instructions::Vector::vmovImmediate(C21_Register, 0, Instructions::I32));
    backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
    backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C21_Register));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
    backend.addInstruction(Instructions::Vector::vmovRegister(C01_Register, C21_Register));
    backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
    backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C21_Register));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
    backend.addInstruction(Instructions::Vector::vmovRegister(C11_Register, C21_Register));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
    backend.addInstruction(Instructions::Vector::vmovRegister(C20_Register, C21_Register));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
    backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));

    backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
    
    // load next b values before loop start
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
    
    // start k loop
    backend.addInstruction(Instructions::Base::dls(Instructions::R6));

    Instructions::Instruction16 * kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
    backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
    backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
    backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));

    backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));
    backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, Instructions::R3, 3));

    // loop end
    backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);

    // last iteration (out of k loop)
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
    backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_Pointer, ldc*4 + 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc*8));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
    backend.addInstruction(Instructions::Vector::vstrw(C21_Register, C_Pointer, ldc*8 + 16));

    // prepare for next i loop. execute earlier to access new i value
    backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));


    // Rewind
    // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
    backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
    // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
    // Rewind B => B = B - 4*k
    backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
    // Rewind C => C += 8*4
    backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 8*4));

    // ensure that only full microkernels can be executed
    backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % 8)));
    backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT);

    // handle i loop edge cases
    if (m % 8 != 0) {
        // we can use two vector registers if m % 8 > 4 (else use only 1 vector register)
        // if two vector registers for A are used, then an predicated 8x3 microkernel is employed
        // only certain instructions are predicated as only those are important for the validity:
        //      - last load
        //      - all stores

        // if one vector register is used, try to employ an (if needed predicated) 4x7 microkernel
        if (m % 8 > 4) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
            // load b[2len]
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 8*ldb));
            // load b[0] and write back
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

            // init accumulators
            backend.addInstruction(Instructions::Vector::vmovImmediate(C21_Register, 0, Instructions::I32));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C21_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C01_Register, C21_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C21_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C11_Register, C21_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C20_Register, C21_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));

            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
            
            // load next b values before loop start
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
            
            // start k loop
            backend.addInstruction(Instructions::Base::dls(Instructions::R6));

            Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));

            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));
            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, Instructions::R3, 3));

            // loop end
            backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

            // last iteration (out of k loop)
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc*8));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
            backend.addInstruction(Instructions::Vector::vpst(4));
            backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
            backend.addInstruction(Instructions::Vector::vpst(3));
            backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_Pointer, ldc*4 + 16));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C21_Register, A1_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C21_Register, C_Pointer, ldc*8 + 16));

            // prepare for next i loop. execute earlier to access new i value
            // backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));


            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            // Rewind B => B = B - 4*k
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
            // Rewind C => C += 8*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (m%8)*4));
        } else if (m % 8 == 4) {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
            // load b[2len]
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 8*ldb));
            // load b[0] and write back
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

            // init accumulators
            backend.addInstruction(Instructions::Vector::vmovImmediate(C20_Register, 0, Instructions::I32));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C20_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C20_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
            
            // load next b values before loop start
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
            
            // start k loop
            backend.addInstruction(Instructions::Base::dls(Instructions::R6));

            Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));
            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, Instructions::R3, 3));

            // loop end
            backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

            // last iteration (out of k loop)
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc*8));

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            // Rewind B => B = B - 4*k
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
            // Rewind C => C += 4*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 4*4));
        } else {
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
            // load b[2len]
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, 8*ldb));
            // load b[0] and write back
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

            // init accumulators
            backend.addInstruction(Instructions::Vector::vmovImmediate(C20_Register, 0, Instructions::I32));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C20_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C20_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            // backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
            
            // load next b values before loop start
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
            
            // start k loop
            backend.addInstruction(Instructions::Base::dls(Instructions::R6));
            Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));

            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));
            backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B2_Register, B_Pointer, Instructions::R3, 3));

            // loop end
            backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

            // last iteration (out of k loop)
            backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
            backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
            backend.addInstruction(Instructions::Vector::vpst(4));
            backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
            backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
            backend.addInstruction(Instructions::Vector::vpst(3));
            backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
            backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
            backend.addInstruction(Instructions::Vector::vstrw(C20_Register, C_Pointer, ldc*8));

            // Rewind
            // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
            // Rewind B => B = B - 4*k
            backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
            // Rewind C => C += 4*4
            backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, (m % 8) * 4));
        }
    }

    // gemm loop i end (next j)
    // Rewind A -> already rewinded by i so need to reset
    backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
    // Rewind B -> rewinded by i to start. add 3*len
    backend.addInstruction(Instructions::Arithmetic::addImmediate32(B_Pointer, ldb*3*4));
    // Rewind C -> still have to go two lines -> 2ldc
    backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 2*ldc*4));

    backend.addInstruction(Instructions::Arithmetic::addImmediate32(J_Loop_Register, 3));
    backend.addInstruction(Instructions::Base::cmpImmediate16(J_Loop_Register, n - (n % 3)));
    backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);

    // handle j loop edge cases
    if (n % 3 == 2) { // 8x2 kernel
        /*
        * Loop i (m loop): Count from 0 to m
        */
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));
        Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
        // load b[0] and write back
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

        // init accumulators
        backend.addInstruction(Instructions::Vector::vmovImmediate(C11_Register, 0, Instructions::I32));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C11_Register));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vmovRegister(C01_Register, C11_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C11_Register));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
        
        // load next b values before loop start
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
        
        // start k loop
        backend.addInstruction(Instructions::Base::dls(Instructions::R6));

        Instructions::Instruction16 * kLoopStartjTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

        backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));

        // loop end
        backend.addLowOverheadBranchFromCurrentPosition(kLoopStartjTail);

        // last iteration (out of k loop)
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_Pointer, ldc*4 + 16));

        // prepare for next i loop. execute earlier to access new i value
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));


        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // backend.addInstruction(Instructions::DataProcessing::movRegister32(A_Pointer, A_Base_Pointer));
        // Rewind B => B = B - 4*k
        backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
        // Rewind C => C += 8*4
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 8*4));

        // ensure that only full microkernels can be executed
        backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % 8)));
        backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

        if (m % 8 != 0) {
            if (m % 8 > 4) {
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C21_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C01_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C11_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                
                // load next b values before loop start
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));

                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

                backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(4));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C11_Register, A1_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(1));
                backend.addInstruction(Instructions::Vector::vstrw(C11_Register, C_Pointer, ldc*4 + 16));
            } else if (m % 8 == 4) {
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C20_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C20_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C20_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                
                // load next b values before loop start
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));

                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

                backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
            } else {
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C20_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C20_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C10_Register, C20_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                // backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                
                // load next b values before loop start
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));
                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));

                backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(4));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(1));
                backend.addInstruction(Instructions::Vector::vstrw(C10_Register, C_Pointer, ldc*4));
            }
        }
    }
    if (n % 3 == 1) {
        backend.addInstruction(Instructions::DataProcessing::movImmediate32(I_Loop_Register, 0));
        Instructions::Instruction16 * iLoopStartjTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, 4*ldb));
        // load b[0] and write back
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

        // init accumulators
        backend.addInstruction(Instructions::Vector::vmovImmediate(C01_Register, 0, Instructions::I32));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
        backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C01_Register));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
        
        // load next b values before loop start
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B1_Register, B_Pointer, ldb*4));
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B2_Register, B_Pointer, ldb*8));
        
        // start k loop
        backend.addInstruction(Instructions::Base::dls(Instructions::R6));

        Instructions::Instruction16 * kLoopStartjTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
        backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

        backend.addInstruction(Instructions::DataProcessing::ldrRegister32(B1_Register, B_Pointer, Instructions::R3, 2));

        // loop end
        backend.addLowOverheadBranchFromCurrentPosition(kLoopStartjTail);

        // last iteration (out of k loop)
        backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
        backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));
        backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
        backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));

        // prepare for next i loop. execute earlier to access new i value
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));


        // Rewind
        // no real rewind. calculate a[i] and restore from base pointer because rewind overflows the immediate
        backend.addInstruction(Instructions::Arithmetic::addRegister32(A_Pointer, A_Base_Pointer, I_Loop_Register, Instructions::LSL, 2));
        // Rewind B => B = B - 4*k
        backend.addInstruction(Instructions::Arithmetic::subImmediate32(B_Pointer, 4*k));
        // Rewind C => C += 8*4
        backend.addInstruction(Instructions::Arithmetic::addImmediate32(C_Pointer, 8*4));

        // ensure that only full microkernels can be executed
        backend.addInstruction(Instructions::Base::cmpImmediate16(I_Loop_Register, m - (m % 8)));
        backend.addBackwardsBranchFromCurrentPosition(iLoopStartjTail, Instructions::LT);

        if (m % 8 != 0) {
            if (m % 8 > 4) {
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C21_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vmovRegister(C00_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vmovRegister(C01_Register, C21_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));

                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C20_Register, A0_Register, B2_Register));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(3));
                backend.addInstruction(Instructions::Vector::vldrw(A1_Register, A_Pointer, 16));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C01_Register, A1_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C01_Register, C_Pointer, 16));

                // prepare for next i loop. execute earlier to access new i value
                // backend.addInstruction(Instructions::Arithmetic::addImmediate32(I_Loop_Register, 8));
            } else if (m % 8 == 4) {
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C00_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));

                //Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C10_Register, A0_Register, B1_Register));
                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
            } else {
                // load b[0] and write back
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));

                // init accumulators
                backend.addInstruction(Instructions::Vector::vmovImmediate(C00_Register, 0, Instructions::I32));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                // backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer, lda*4));

                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));
                
                // load next b values before loop start
                
                // start k loop
                backend.addInstruction(Instructions::Base::dls(Instructions::R6));
                Instructions::Instruction16 * kLoopStartTail = backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Arithmetic::addImmediate32(A_Pointer, lda*4));

                // loop end
                backend.addLowOverheadBranchFromCurrentPosition(kLoopStartTail);

                // last iteration (out of k loop)
                backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(B0_Register, B_Pointer, 4, false, true));
                backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R11, m % 4));
                backend.addInstruction(Instructions::Vector::vctp(Instructions::Size32, Instructions::R11));
                backend.addInstruction(Instructions::Vector::vpst(3));
                backend.addInstruction(Instructions::Vector::vldrw(A0_Register, A_Pointer));
                backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(C00_Register, A0_Register, B0_Register));
                backend.addInstruction(Instructions::Vector::vstrw(C00_Register, C_Pointer));
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
