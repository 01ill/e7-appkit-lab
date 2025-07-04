#include "Backend.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Vector.hpp"
#include <cstdint>

using namespace JIT::Instructions;

void JIT::Backend::addInstruction(Instruction16 instruction) {
    instructions[instructionCount++] = instruction;
}

void JIT::Backend::addInstruction(Instruction32 instruction) {
    instructions[instructionCount++] = static_cast<Instruction16>(instruction >> 16U); // select 16 highest bits
    instructions[instructionCount++] = static_cast<Instruction16>(instruction); // select 16 lowest bits
}

void JIT::Backend::addHeliumInstruction(Instruction32 instruction) {
    if (reinterpret_cast<uintptr_t>(&instructions[instructionCount]) % 4 != 0) { // if not word aligned
        addInstruction(Base::nop16());
    }
    addInstruction(instruction);
}

Instruction16 * JIT::Backend::addBranchTargetInstruction(Instruction32 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 2]; // eingefügte Instruction war 32 Bit = 2 16 Bit Instruktions lang
}

Instruction16 * JIT::Backend::addBranchTargetInstruction(Instruction16 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 1];
}

Instruction16 * JIT::Backend::addBranchPlaceholder(bool shortBranch) {
    return shortBranch ? addBranchTargetInstruction(Base::nop16()) : addBranchTargetInstruction(Base::nop32());
}

void JIT::Backend::addLowOverheadBranchFromCurrentPosition(Instruction16 * loopStart, bool letp) {
    if (letp) {
        addInstruction(Base::letp(getBranchOffset(loopStart) - 4));
    } else {
        addInstruction(Base::le(getBranchOffset(loopStart) - 4));
    }
}

void JIT::Backend::addBackwardsBranchFromCurrentPosition(Instruction16 * branchTarget, Condition branchCondition) {
    int16_t branchOffset = getBranchOffset(branchTarget) - 4;
    if (branchCondition == AL && branchOffset < 2046 && branchOffset > -2048) { // use unconditional 16bit branch
        addInstruction(Base::b16(branchOffset));
    } else if (branchCondition == AL && (branchOffset > 2046 || branchOffset < -2048)) { // use unconditional 32bit branch
        addInstruction(Base::b32(branchOffset));
    } else if (branchCondition != AL && branchOffset < 254 && branchOffset > -256) { // use conditional 16bit branch
        addInstruction(Base::bCond16(branchCondition, branchOffset));
    } else { // use conditional 32bit branch
        addInstruction(Base::bCond32(branchCondition, branchOffset));
    }
}

void JIT::Backend::setForwardsBranch(Instruction16 * branchInstruction, Instruction16 * branchTarget, Condition branchCondition) {
    int16_t branchOffset = (branchTarget - branchInstruction) * 2 - 4;
    if (*branchInstruction == Base::nop16()) { // can only use 16bit branch instruction
        if (branchCondition == AL) {
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
            *branchInstruction = Base::bCond16(branchCondition, branchOffset);
        }
    } else if ((static_cast<Instruction32>(*branchInstruction << 16)) + *(branchInstruction+1) == Base::nop32() ) { // can use 32bit branch instruction
        Instruction32 instr;
        if (branchCondition == AL) {
            instr = Base::b32(branchOffset);
        } else {
            instr = Base::bCond32(branchCondition, branchOffset);
        }
        *branchInstruction = instr >> 16;
        *(branchInstruction+1) = 0xffff & instr;
    } else {
        Base::printValidationError("setForwardsBranch: can only set if placeholder nop is set - returning");
    }
}

int16_t JIT::Backend::getBranchOffset(Instruction16 * instrStart) {
    return (instrStart - &instructions[instructionCount]) * 2; // jede Instruktion sind 16 Bit = 2 Byte
    //return &instructions[instructionCount] - instrStart;
}

void JIT::Backend::predicateNextInstructions(uint32_t countInstructions) {
    maxPredicateInstructions = countInstructions;
    predicateCounter = 0;
    addInstruction(Vector::vpst(countInstructions));
}

void JIT::Backend::insertPredicatedInstruction(Instructions::Instruction32 instr) {
    if (maxPredicateInstructions == -1) { // no predication used
        addInstruction(instr);
    } else if (predicateCounter >= maxPredicateInstructions) {
        Base::printValidationError("insertPredicatedInstruction: all predicated spots have been used - inserting nop");
        addInstruction(Base::nop32());
    } else {
        predicateCounter++;
        addInstruction(instr);
    }
}

void JIT::Backend::clearPredication() {
    maxPredicateInstructions = -1;
    predicateCounter = -1;
}

Instruction16 * JIT::Backend::getInstructions() {
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