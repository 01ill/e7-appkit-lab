#include "generators/PeakPerformance.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cassert>
#include <cstdint>

void (*JIT::Generators::PeakPerformance::generate(uint32_t operational_intensity)) (uint32_t size) {
    backend.resetKernel();

    // push {r4, lr}
    backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
    backend.addInstruction(Instructions::Base::nop16());

    // dls
    backend.addInstruction(Instructions::Base::dls(Instructions::Register::R0));

    //        vfma.f32 q0, q2, q1
    Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Vector::vfma(Instructions::Q0, Instructions::Q2, Instructions::Q1));
    //        vfma.f32 q1, q3, q2
    backend.addInstruction(Instructions::Vector::vfma(Instructions::Q1, Instructions::Q3, Instructions::Q2));
    //        vfma.f32 q2, q0, q3
    backend.addInstruction(Instructions::Vector::vfma(Instructions::Q2, Instructions::Q0, Instructions::Q3));
    //        vfma.f32 q3, q1, q0
    backend.addInstruction(Instructions::Vector::vfma(Instructions::Q3, Instructions::Q1, Instructions::Q0));

    // generate according to operational intensity
    for (uint32_t i = 1; i < operational_intensity; i++) {
        //        vfma.f32 q0, q2, q1
       backend.addInstruction(Instructions::Vector::vfma(Instructions::Q0, Instructions::Q2, Instructions::Q1));
        //        vfma.f32 q1, q3, q2
        backend.addInstruction(Instructions::Vector::vfma(Instructions::Q1, Instructions::Q3, Instructions::Q2));
        //        vfma.f32 q2, q0, q3
        backend.addInstruction(Instructions::Vector::vfma(Instructions::Q2, Instructions::Q0, Instructions::Q3));
        //        vfma.f32 q3, q1, q0
        backend.addInstruction(Instructions::Vector::vfma(Instructions::Q3, Instructions::Q1, Instructions::Q0));
    }

    // le lr, -> branch to dlstpStart
    backend.addLowOverheadBranchFromCurrentPosition(dlstpStart);

    // pop {pc}
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

    backend.clearCaches();
    return reinterpret_cast<Func>(backend.getThumbAddress());
}

void (*JIT::Generators::PeakPerformance::generateVfma(uint32_t vfmaCount)) () {
    backend.resetKernel();

    // push {r4, lr}
    backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));

    for (uint32_t i = 0; i < vfmaCount; i++)
        backend.addInstruction(Instructions::Vector::vfma(Instructions::Q0, Instructions::Q2, Instructions::Q1));

    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

    backend.clearCaches();
    return reinterpret_cast<FuncVoid>(backend.getThumbAddress());
}


// void (*JIT::Generators::PeakPerformance::generateNoMem(uint32_t operational_intensity)) (float const * a, float const * b, float * c, uint32_t size) {
//     backend.resetKernel();

//     // push {lr}
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
//     // push r4
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::R4));

//     // dlstp
//     Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Base::dlstp(Instructions::Register::R3, Instructions::Size32));

//     // backend.addInstruction(Instructions::Base::nop());
//     // vldrw.f32 q0, [r0]
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 0, false, false));
//     // vldrw.f32 q0, [r1]
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R1, 0, false, false));

//     // generate according to operational intensity
//     for (uint32_t i = 0; i < operational_intensity; i++) {
//         //        vfma.f32 q0, q2, q1
//        backend.addInstruction(Instructions::Vector::vfma(Instructions::Q0, Instructions::Q2, Instructions::Q1, 0));
//         //        vfma.f32 q1, q3, q2
//         backend.addInstruction(Instructions::Vector::vfma(Instructions::Q1, Instructions::Q3, Instructions::Q2, 0));
//         //        vfma.f32 q2, q0, q3
//         backend.addInstruction(Instructions::Vector::vfma(Instructions::Q2, Instructions::Q0, Instructions::Q3, 0));
//         //        vfma.f32 q3, q1, q0
//         backend.addInstruction(Instructions::Vector::vfma(Instructions::Q3, Instructions::Q1, Instructions::Q0, 0));
//     }

//     // vstrw.f32 q2, [r2]
//     // backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, 0, false, false, false));

//     // letp lr, -> branch to dlstpStart
//     backend.addInstruction(Instructions::Base::letp(backend.getBranchOffset(dlstpStart) + 2));

//     // pop r4
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4));
//     // pop {pc}
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

//     __asm("dsb");
//     __asm("isb");

//     return reinterpret_cast<Func>(backend.getThumbAddress());
// }

// void (*JIT::Generators::PeakPerformance::generateSteps(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size) {
//     backend.resetKernel();

//     // push {lr}
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
//     // push r4
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::R4));

//     // dlstp
//     Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Base::dlstp(Instructions::Register::R3, Instructions::Size32));

//     // backend.addInstruction(Instructions::Base::nop());
//     // vldrw.f32 q0, [r0], #16
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 4, false, true));
//     // vldrw.f32 q0, [r1], #16
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R1, 4, false, true));
//     uint32_t count = 0;
//     // generate according to operational intensity
//     for (float i = 0; i < operational_intensity; i += 0.25) {
//         backend.addInstruction(Instructions::Vector::vfma(static_cast<Instructions::VectorRegister>(count % 4), static_cast<Instructions::VectorRegister>((count + 2) % 4), static_cast<Instructions::VectorRegister>((count + 1) % 4), 0));
//         count++;
//     }

//     // vstrw.f32 q2, [r2]
//     // backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, 4, false, true, false));

//     // letp lr, -> branch to dlstpStart
//     backend.addInstruction(Instructions::Base::letp(backend.getBranchOffset(dlstpStart) + 2));

//     // pop r4
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4));
//     // pop {pc}
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

//     __asm("dsb");
//     __asm("isb");

//     return reinterpret_cast<Func>(backend.getThumbAddress());
// }

// void (*JIT::Generators::PeakPerformance::generateStepsNoMem(float operational_intensity)) (float const * a, float const * b, float * c, uint32_t size) {
//     backend.resetKernel();

//     // push {lr}
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
//     // push r4
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::R4));

//     // dlstp
//     Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Base::dlstp(Instructions::Register::R3, Instructions::Size32));

//     // backend.addInstruction(Instructions::Base::nop());
//     // vldrw.f32 q0, [r0]
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 0, false, false));
//     // vldrw.f32 q0, [r1]
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R1, 0, false, false));

//     uint32_t count = 0;
//     // generate according to operational intensity
//     for (float i = 0; i < operational_intensity; i += 0.25) {
//         backend.addInstruction(Instructions::Vector::vfma(static_cast<Instructions::VectorRegister>(count % 4), static_cast<Instructions::VectorRegister>((count + 2) % 4), static_cast<Instructions::VectorRegister>((count + 1) % 4), 0));
//         count++;
//     }

//     // vstrw.f32 q2, [r2]
//     // backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, 0, false, false, false));

//     // letp lr, -> branch to dlstpStart
//     backend.addInstruction(Instructions::Base::letp(backend.getBranchOffset(dlstpStart) + 2));

//     // pop r4
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4));
//     // pop {pc}
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

//     __asm("dsb");
//     __asm("isb");

//     return reinterpret_cast<Func>(backend.getThumbAddress());
// }

// void (*JIT::Generators::PeakPerformance::generate(uint32_t flopsPerByte, uint32_t vectorCount)) (float const * a, float const * b, float * c, uint32_t size) {
//     backend.resetKernel();

//     // push {lr}
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
//     // push r4
//     backend.addInstruction(Instructions::DataProcessing::push32(Instructions::R4));

//     // dlstp
//     Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Base::dlstp(Instructions::Register::R3, Instructions::Size32));

//     // vldrw.f32 q0, [r0], #16
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 4, false, true));
//     // vldrw.f32 q0, [r1], #16
//     backend.addInstruction(Instructions::Vector::vldrw(Instructions::Q1, Instructions::R1, 4, false, true)); // TODO

//     assert(vectorCount % 2 == 0);
//     assert(vectorCount <= 8);
//     for (uint32_t i = 0; i < vectorCount; i += 2) {
//         backend.addInstruction(Instructions::Vector::vldrw(static_cast<Instructions::VectorRegister>(i), Instructions::R0, 4, false, true));
//         backend.addInstruction(Instructions::Vector::vldrw(static_cast<Instructions::VectorRegister>(i+1), Instructions::R1, 4, false, true));
//     }

//     // 1 vector = 128 Bit = 16 Byte i.e. for each vector we need flopsPerByte * 16 flops and we progress in 2 vector steps
//     // each vfma instruction has 8 flops, so we need 2 vfma instructions for each vector so for an 2 vector step we need 4 instructions per flopsPerByte
//     // Vector Count = 2 ==> 32 Byte ==> 4 VFMA instructions

//     // generate according to flopsPerByte
//     uint32_t vfmaCount = vectorCount * flopsPerByte * 2;
//     for (uint32_t i = 0; i < vfmaCount; i++) {
//         backend.addInstruction(Instructions::Vector::vfma(
//             static_cast<Instructions::VectorRegister>((i + 2) % vectorCount), 
//             static_cast<Instructions::VectorRegister>((i) % vectorCount),
//             static_cast<Instructions::VectorRegister>((i + 1) % vectorCount),
//             false
//         ));
//     }

//     // vstrw.f32 q2, [r2]
//     // backend.addInstruction(Instructions::Vector::vstrw(Instructions::Q2, Instructions::R2, 4, false, true, false));

//     // letp lr, -> branch to dlstpStart
//     backend.addInstruction(Instructions::Base::letp(backend.getBranchOffset(dlstpStart) + 2));

//     // pop r4
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::R4));
//     // pop {pc}
//     backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

//     __asm("dsb");
//     __asm("isb");

//     return reinterpret_cast<Func>(backend.getThumbAddress());
// }
