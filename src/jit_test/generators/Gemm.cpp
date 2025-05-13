#include "Gemm.hpp"
#include "instructions/Arithmetic.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cstdint>


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
    backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(Instructions::R10, Instructions::R0)); // save a pointer
    backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(Instructions::R11, Instructions::R1)); // save b pointer
    backend.addInstruction(JIT::Instructions::DataProcessing::movRegister32(Instructions::R12, Instructions::R2)); // save c pointer
    backend.addInstruction(JIT::Instructions::DataProcessing::movImmediate32(Instructions::R4, 0)); // mov 0 to r4
    // backend.addInstruction(Instructions::Base::nop16());
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R6, k-2));
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R3, ldb));

    /*
    * Loop j (n loop): Count from 0 to n
    */
    // Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(JIT::Instructions::Base::cmpImmediate16(Instructions::R4, n));

    // Instructions::Instruction16 * jLoopEndBGE = backend.addBranchTargetInstruction(Instructions::Base::nop16());
    Instructions::Instruction16 * jLoopStart = backend.addBranchTargetInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R5, 0));


    /*
    * Loop i (m loop): Count from 0 to m
    */
    // Instructions::Instruction16 * iLoopStart = backend.addBranchTargetInstruction(JIT::Instructions::Base::cmpImmediate16(Instructions::R5, m));
    // Instructions::Instruction16 * iLoopEndBGE = backend.addBranchTargetInstruction(Instructions::Base::nop16());

    /* Calculate tile pointers */
    // a[i]
    Instructions::Instruction16 * iLoopStart = backend.addBranchTargetInstruction(Instructions::Arithmetic::addRegister32(Instructions::R0, Instructions::R10, Instructions::R5, Instructions::LSL, 2));
    // j * ldb
    backend.addInstruction(Instructions::Arithmetic::mul32(Instructions::R2, Instructions::R4, Instructions::R3));
    // b[j * ldb]
    backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R1, Instructions::R11, Instructions::R2, Instructions::LSL, 2));
    // j * ldc
    backend.addInstruction(Instructions::DataProcessing::movImmediate32(Instructions::R3, ldc));
    backend.addInstruction(Instructions::Arithmetic::mul32(Instructions::R2, Instructions::R4, Instructions::R3));
    // c[j * ldc + i]
    backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R2, Instructions::R12, Instructions::R2, Instructions::LSL, 2));
    backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R2, Instructions::R2, Instructions::R5, Instructions::LSL, 2));
    // load b[len]
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R7, Instructions::R1, 4*ldb));
    // load b[2len]
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R8, Instructions::R1, 8*ldb));
    // load b[0] and write back
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R9, Instructions::R1, 4, false, true));

    // init accumulators
    backend.addInstruction(Instructions::Vector::vmovImmediate(Instructions::Q5, 0, Instructions::I32));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q6, Instructions::R0));
    backend.addInstruction(Instructions::Vector::vmovRegister(Instructions::Q0, Instructions::Q5));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q0, Instructions::Q6, Instructions::R9));
    backend.addInstruction(Instructions::Vector::vmovRegister(Instructions::Q1, Instructions::Q5));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q7, Instructions::R0, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q1, Instructions::Q7, Instructions::R9));
    backend.addInstruction(Instructions::Vector::vmovRegister(Instructions::Q2, Instructions::Q5));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q2, Instructions::Q6, Instructions::R7));
    backend.addInstruction(Instructions::Vector::vmovRegister(Instructions::Q3, Instructions::Q5));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q3, Instructions::Q7, Instructions::R7));
    backend.addInstruction(Instructions::Vector::vmovRegister(Instructions::Q4, Instructions::Q5));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q4, Instructions::Q6, Instructions::R8));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q6, Instructions::R0, lda*4));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q5, Instructions::Q7, Instructions::R8));

    backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R0, Instructions::R0, Instructions::R3, Instructions::LSL, 2));
    

    // load next a[0]
    // try using immediate
    /*if (lda*4 < 508) {
    } else {
        backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R0, Instructions::R0, Instructions::R3, Instructions::LSL, 2));
        backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q6, Instructions::R0));
    }*/

    // load next b values before loop start
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R7, Instructions::R1, ldb*4));
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R8, Instructions::R1, ldb*8));
    
    // start k loop
    backend.addInstruction(Instructions::Base::dls(Instructions::R6));

    Instructions::Instruction16 * kLoopStart = backend.addBranchTargetInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q2, Instructions::Q6, Instructions::R7));
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R9, Instructions::R1, 4, false, true));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q4, Instructions::Q6, Instructions::R8));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q0, Instructions::Q6, Instructions::R9));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q7, Instructions::R0, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q1, Instructions::Q7, Instructions::R9));
    backend.addInstruction(Instructions::Arithmetic::addRegister32(Instructions::R0, Instructions::R0, Instructions::R3, Instructions::LSL, 2));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q3, Instructions::Q7, Instructions::R7));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q6, Instructions::R0));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q5, Instructions::Q7, Instructions::R8));

    backend.addInstruction(Instructions::DataProcessing::ldrRegister32(Instructions::R7, Instructions::R1, Instructions::R3, 2));
    backend.addBranchTargetInstruction(Instructions::DataProcessing::ldrRegister32(Instructions::R8, Instructions::R1, Instructions::R3, 3));

    // loop end
    backend.addLowOverheadBranchFromCurrentPosition(kLoopStart);

    // last iteration (out of k loop)
    backend.addInstruction(Instructions::DataProcessing::ldrImmediate32(Instructions::R9, Instructions::R1, 4, false, true));
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q7, Instructions::R0, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q1, Instructions::Q7, Instructions::R9));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q1, Instructions::R2, 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q0, Instructions::Q6, Instructions::R9));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q0, Instructions::R2));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q2, Instructions::Q6, Instructions::R7));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, ldc*4));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q3, Instructions::Q7, Instructions::R7));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q3, Instructions::R2, ldc*4 + 16));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q4, Instructions::Q6, Instructions::R8));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q4, Instructions::R2, ldc*8));
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q5, Instructions::Q7, Instructions::R8));
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q5, Instructions::R2, ldc*8 + 16));


    // Rewind
    // Rewind A => A = A - k*lda + 8 ==> A -= k*lda -8
    backend.addInstruction(Instructions::Arithmetic::subImmediate32(Instructions::R0, k*lda - 8));



    backend.addInstruction(Instructions::Arithmetic::addImmediate32(Instructions::R5, 8));
    backend.addInstruction(Instructions::Base::cmpImmediate16(Instructions::R5, m));

    // backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::AL);
    backend.addBackwardsBranchFromCurrentPosition(iLoopStart, Instructions::LT);

    // gemm loop i end
    Instructions::Instruction16 * iLoopEnd = backend.addBranchTargetInstruction(Instructions::Arithmetic::addImmediate32(Instructions::R4, 3));
    backend.addInstruction(Instructions::Base::cmpImmediate16(Instructions::R4, n));
    backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::LT);
    // backend.addBackwardsBranchFromCurrentPosition(jLoopStart, Instructions::AL);

    // gemm loop j end
    Instructions::Instruction16 * jLoopEnd = backend.addBranchTargetInstruction(Instructions::DataProcessing::vpop(Instructions::Q4, 4));
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4, Instructions::R5, Instructions::R6, Instructions::R7, Instructions::R8, Instructions::R9, Instructions::R10, Instructions::R11, Instructions::R12, Instructions::PC));

    // now insert the correct branch instructions after cmp
    //backend.setForwardsBranch(iLoopEndBGE, iLoopEnd, Instructions::GE);
    //backend.setForwardsBranch(jLoopEndBGE, jLoopEnd, Instructions::GE);


    __asm("dsb");
    __asm("isb");
    return reinterpret_cast<Func>(backend.getThumbAddress());
}
