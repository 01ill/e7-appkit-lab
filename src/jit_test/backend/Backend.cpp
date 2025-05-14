#include "Backend.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

using namespace JIT::Instructions;

void JIT::Backend::addInstruction(Instructions::Instruction16 instruction) {
    instructions[instructionCount++] = instruction;
}

void JIT::Backend::addInstruction(Instructions::Instruction32 instruction) {
    instructions[instructionCount++] = static_cast<Instructions::Instruction16>(instruction >> 16U); // select 16 highest bits
    instructions[instructionCount++] = static_cast<Instructions::Instruction16>(instruction); // select 16 lowest bits
}

void JIT::Backend::addHeliumInstruction(Instructions::Instruction32 instruction) {
    if (reinterpret_cast<uintptr_t>(&instructions[instructionCount-1]) % 4 != 0) { // if not word aligned
        addInstruction(Instructions::Base::nop16());
    }
    addInstruction(instruction);
}

JIT::Instructions::Instruction16 * JIT::Backend::addBranchTargetInstruction(Instructions::Instruction32 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 2]; // eingefügte Instruction war 32 Bit = 2 16 Bit Instruktions lang
}

JIT::Instructions::Instruction16 * JIT::Backend::addBranchTargetInstruction(Instructions::Instruction16 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 1];
}

Instruction16 * JIT::Backend::addBranchPlaceholder(bool longBranch) {
    return longBranch ? addBranchTargetInstruction(Base::nop32()) : addBranchTargetInstruction(Base::nop16());
}

void JIT::Backend::addLowOverheadBranchFromCurrentPosition(Instructions::Instruction16 * loopStart, bool letp) {
    if (letp) {
        addInstruction(Instructions::Base::letp(getBranchOffset(loopStart)-4));
    } else {
        addInstruction(Instructions::Base::le(getBranchOffset(loopStart)-4));
    }
}

void JIT::Backend::addBackwardsBranchFromCurrentPosition(Instructions::Instruction16 * branchTarget, Instructions::Condition branchCondition) {
    int16_t branchOffset = getBranchOffset(branchTarget);
    if (branchCondition == Instructions::AL && branchOffset < 2046 && branchOffset > -2048) { // use unconditional 16bit branch
        addInstruction(Base::b16(branchOffset-4));
    } else if (branchCondition == Instructions::AL && (branchOffset > 2046 || branchOffset < -2048)) { // use unconditional 32bit branch
        addInstruction(Base::b32(branchOffset-4));
    } else if (branchCondition != AL && branchOffset < 254 && branchOffset > -256) { // use conditional 16bit branch
        addInstruction(Base::bCond16(branchCondition, branchOffset-4));
    } else { // use conditional 32bit branch
        addInstruction(Base::bCond32(branchCondition, branchOffset-4));
    }
}

void JIT::Backend::setForwardsBranch(Instructions::Instruction16 * branchInstruction, Instructions::Instruction16 * branchTarget, Instructions::Condition branchCondition) {
    int16_t branchOffset = (branchTarget - branchInstruction) * 2;
    if (*branchInstruction == Base::nop16()) { // can only use 16bit branch instruction
        if (branchCondition == Instructions::AL) {
            if (branchOffset > 2046 || branchOffset < -2048) {
                Base::printValidationError("setForwardsBranch: 16bit branch only supports branch from [-2048, 2046] - please replace nop16 with nop32");
                return;
            }
            *branchInstruction = Base::b16(branchOffset);
        } else {
            if (branchOffset > 254 || branchOffset < -256) {
                Base::printValidationError("setForwardsBranch: 16bit branch only supports branch from [-256, 254] - please replace nop16 with nop32");
                return;
            }
            *branchInstruction = Base::bCond16(branchCondition, branchOffset-4);
        }
    } else if (*branchInstruction == Base::nop32()) { // can use 32bit branch instruction

    } else {
        Base::printValidationError("setForwardsBranch: can only set if placeholder nop is set - returning");
    }
}


int16_t JIT::Backend::getBranchOffset(Instructions::Instruction16 * instrStart) {
    return (instrStart - &instructions[instructionCount]) * 2; // jede Instruktion sind 16 Bit = 2 Byte
    //return &instructions[instructionCount] - instrStart;
}

JIT::Instructions::Instruction16 * JIT::Backend::getInstructions() {
    return instructions;
}

uint16_t JIT::Backend::getInstructionCount() {
    return instructionCount;
}

uintptr_t JIT::Backend::getThumbAddress() const {
    //return reinterpret_cast<uintptr_t>(instructions);
    return reinterpret_cast<uintptr_t>(instructions) | 0x1U;
}

void JIT::Backend::resetKernel() {
    // as no dynamic memory allocation is used, it is sufficient to just reset the instruction count/pointer
    instructionCount = 0;
}