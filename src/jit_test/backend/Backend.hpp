#ifndef BACKEND_HPP
#define BACKEND_HPP
#pragma once
#include <cstdint>
#include <cstring>
#include "../instructions/Base.hpp"

namespace JIT {
    class Backend;
}


class JIT::Backend {
    public:
        Backend(Instructions::Instruction16 * instructionBuffer, uint32_t bufferSize) {
            this->instructions = instructionBuffer;
            this->maxInstructionCount = bufferSize;
        }

        /**
         * @brief Add 16-bit instruction to the code buffer
         * 
         * @param instruction encoded instruction
         */
        void addInstruction(Instructions::Instruction16 instruction);
        /**
         * @brief Add 32-bit instruction to the code buffer
         * 
         * @param instruction encoded instruction
         */
        void addInstruction(Instructions::Instruction32 instruction);
        /// @brief Adds a Helium instruction to the kernel and aligns it to 32Bit.
        /// @details ARM recommends to align Helium instructions to 32Bit. Otherwise, there might be a performance hit
        /// @see https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/armv8_2d00_m-based-processor-software-development-hints-and-tips
        /// @param instruction 
        void addHeliumInstruction(Instructions::Instruction32 instruction);
        
        Instructions::Instruction16* addBranchTargetInstruction(Instructions::Instruction16 branchInstruction);
        Instructions::Instruction16* addBranchTargetInstruction(Instructions::Instruction32 branchInstruction);
        Instructions::Instruction16* addBranchPlaceholder(bool shortBranch = false);
        void addLowOverheadBranchFromCurrentPosition(Instructions::Instruction16 * loopStart, bool letp = false);
        void addBackwardsBranchFromCurrentPosition(Instructions::Instruction16 * branchTarget, Instructions::Condition branchCondition);
        void setForwardsBranch(Instructions::Instruction16 * branchInstruction, Instructions::Instruction16 * branchTarget, Instructions::Condition branchCondition);
        int16_t getBranchOffset(Instructions::Instruction16 * instrStart);
        void insertWlsLabel(Instructions::Instruction16 wlsPosition, int16_t imm12);

        void predicateNextInstructions(uint32_t countInstructions);
        void insertPredicatedInstruction(Instructions::Instruction32 instr);
        void clearPredication();

        uintptr_t getThumbAddress() const;
        Instructions::Instruction16 * getInstructions();
        uint16_t getInstructionCount();
        void resetKernel();
        void copyToBuffer(Instructions::Instruction16 * globalBuffer) const {
	        std::memcpy(globalBuffer, instructions, instructionCount * sizeof(Instructions::Instruction16));
        }
        uintptr_t getBufferThumbAddress(Instructions::Instruction16 * globalBuffer) const {
            return reinterpret_cast<uintptr_t>(globalBuffer) | 0x1U;
        }
        void clearCaches();

    private:
        // Instructions::Instruction16 * instructionBuffer;
        // Instructions::Instruction16 instructions[3072] = {0};
        uint32_t maxInstructionCount = 0;
        Instructions::Instruction16 * instructions;
        uint16_t instructionCount = 0;
        int32_t predicateCounter = 0;
        int32_t maxPredicateInstructions = 0;
};

#endif // BACKEND_HPP