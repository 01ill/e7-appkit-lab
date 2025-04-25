#include "Triad.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cstdint>

void (*JIT::Generators::Triad::generate(uint32_t count)) (float const * a, float const * b, float * c, float const scalar) {
    // push r4
    backend.addInstruction(Instructions::DataProcessing::push(Instructions::R4));
    // push {lr}
    backend.addInstruction(Instructions::DataProcessing::push(Instructions::Register::LR));
    // vmov.f32 r3, s0
    backend.addInstruction(Instructions::Vector::vmovGPxScalar(true, Instructions::S0, Instructions::R3));

    // set count
    backend.addInstruction(Instructions::DataProcessing::mov(Instructions::R4, count));

    // dlstp
    Instructions::Instruction16 * dlstpStart = backend.addBranchInstruction(Instructions::Base::dlstp(Instructions::Register::R4, Instructions::Size32)); // todo
    // vldrw.f32 q0, [r0], #16
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 4, 0, 1, 0));
    // vldrw.f32 q0, [r1], #16
    backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R1, 4, 0, 1, 0));
    // vfma.f32 q2, q0, r3
    backend.addInstruction(Instructions::Vector::vfmaVectorByScalarPlusVector(Instructions::Q2, Instructions::Q0, Instructions::R3, 0));
    // vstrw.f32 q2, [r2], #16
    backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, 4, 0, 1, 0));

    // letp lr, -> branch to dlstpStart
    backend.addInstruction(Instructions::Base::letp(backend.getBranchOffset(dlstpStart) + 2));

    // pop r4
    backend.addInstruction(Instructions::DataProcessing::pop(Instructions::R4));
    // pop {pc}
    backend.addInstruction(Instructions::DataProcessing::pop(Instructions::PC));

    __asm("dsb");
    __asm("isb");

    return reinterpret_cast<Func>(backend.getThumbAddress());
}