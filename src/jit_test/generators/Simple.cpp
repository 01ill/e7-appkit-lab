#include "Simple.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include <cstdint>



uint32_t ( *JIT::Generators::Simple::generate())() {

    // backend.addInstruction(Instructions::DataProcessing::ldr(Instructions::R0, Instructions::R1));

    backend.addInstruction(Instructions::DataProcessing::mov(Instructions::Register::R0, 3));

    backend.addInstruction(Instructions::Base::bx(Instructions::Register::LR));


    __asm("dsb");
    __asm("isb");
    return reinterpret_cast<Func>(backend.getThumbAddress());
}
