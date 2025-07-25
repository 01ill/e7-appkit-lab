#include "catch2/catch_amalgamated.hpp"
#include "backend/Backend.hpp"
#include "instructions/Arithmetic.hpp"
#include "instructions/Base.hpp"
#include "instructions/Vector.hpp"

#include <cstdint>

using namespace JIT;


/**
 * Am Anfang: Half Word aligned oder Word aligned
 * 
 */
TEST_CASE("Helium instructions are correctly aligned", "[BACKEND]") {
    Backend backend;
    Instructions::Instruction16 * instructions = backend.getInstructions();
    if (reinterpret_cast<uintptr_t>(instructions) % 4 == 0) {
        backend.addInstruction(Instructions::Base::nop16());
        REQUIRE(reinterpret_cast<uintptr_t>(&instructions[backend.getInstructionCount()]) % 2 == 0);
    }
    REQUIRE(reinterpret_cast<uintptr_t>(&instructions[backend.getInstructionCount()]) % 4 != 0);
    backend.addHeliumInstruction(Instructions::Vector::vldrw(JIT::Instructions::Q0, JIT::Instructions::R4, 0, 0, 0));
    REQUIRE(reinterpret_cast<uintptr_t>(&instructions[backend.getInstructionCount()]) % 4 == 0);
    Instructions::Instruction16 * heliumStart = &instructions[backend.getInstructionCount() - 2];
    REQUIRE(reinterpret_cast<uintptr_t>(heliumStart) % 4 == 0);
}
/*
TEST_CASE("Branching Operations", "[BRANCH]") {
    SECTION("Low Overhead Branch - Backwards LE/LETP") {
        Backend backend;
        backend.addInstruction(Instructions::Base::dls(Instructions::R0));
        Instructions::Instruction16 * dlsStart = backend.addBranchTargetInstruction(Instructions::Vector::vfma(JIT::Instructions::Q0, JIT::Instructions::Q1, JIT::Instructions::Q2));
        backend.addLowOverheadBranchFromCurrentPosition(dlsStart);
        REQUIRE(&backend.getInstructions()[backend.getInstructionCount()-1] == dlsStart);
    }
}*/