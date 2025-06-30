#include "generators/Throughput.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cassert>
#include <cstdint>

void (*JIT::Generators::Throughput::generate()) (float const *a, uint32_t size) {
    backend.resetKernel();

    // push {r4, lr}
    backend.addInstruction(Instructions::DataProcessing::push32(Instructions::LR));
    backend.addInstruction(Instructions::Base::nop16());

    // dls
    backend.addInstruction(Instructions::Base::dlstp(Instructions::Register::R1, Instructions::Size32));

    Instructions::Instruction16 * dlstpStart = backend.addBranchTargetInstruction(Instructions::Vector::vldrw(Instructions::Q0, Instructions::R0, 16, false, true));

    // le lr, -> branch to dlstpStart
    backend.addLowOverheadBranchFromCurrentPosition(dlstpStart, true);

    // pop {pc}
    backend.addInstruction(Instructions::DataProcessing::pop32(Instructions::PC));

    __asm("dsb");
    __asm("isb");

    return reinterpret_cast<Func>(backend.getThumbAddress());
}
