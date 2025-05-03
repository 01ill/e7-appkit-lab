#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <cstdint>
#include "../instructions/Base.hpp"

namespace JIT {
    class Backend;
}

class JIT::Backend {
    public:
        void addInstruction(Instructions::Instruction16 instruction);
        void addInstruction(Instructions::Instruction32 instruction);
        /// @brief Adds a Helium instruction to the kernel and aligns it to 32Bit.
        /// @details ARM recommends to align Helium instructions to 32Bit. Otherwise, there might be a performance hit
        /// @see https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/armv8_2d00_m-based-processor-software-development-hints-and-tips
        /// @param instruction 
        void addHeliumInstruction(Instructions::Instruction32 instruction);
        Instructions::Instruction16* addBranchInstruction(Instructions::Instruction16 branchInstruction);
        Instructions::Instruction16* addBranchInstruction(Instructions::Instruction32 branchInstruction);
        int16_t getBranchOffset(Instructions::Instruction16 * instrStart);
        void insertWlsLabel(Instructions::Instruction16 wlsPosition, int16_t imm12);
        uintptr_t getThumbAddress() const;
        Instructions::Instruction16 * getInstructions();
        uint16_t getInstructionCount();
        void resetKernel();

    private:
        Instructions::Instruction16 instructions[1024] = {0};
        uint16_t instructionCount = 0;
};

#endif // BACKEND_HPP