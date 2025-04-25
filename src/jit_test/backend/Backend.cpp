#include "Backend.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

void JIT::Backend::addInstruction(Instructions::Instruction16 instruction) {
    instructions[instructionCount++] = instruction;
}

void JIT::Backend::addInstruction(Instructions::Instruction32 instruction) {
    instructions[instructionCount++] = static_cast<Instructions::Instruction16>(instruction >> 16U); // select 16 highest bits
    instructions[instructionCount++] = static_cast<Instructions::Instruction16>(instruction); // select 16 lowest bits
}

JIT::Instructions::Instruction16 * JIT::Backend::addBranchInstruction(Instructions::Instruction32 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 2]; // eingefügte Instruction war 32 Bit = 2 16 Bit Instruktions lang
}

JIT::Instructions::Instruction16 * JIT::Backend::addBranchInstruction(Instructions::Instruction16 branchInstruction) {
    addInstruction(branchInstruction);
    return &instructions[instructionCount - 1];
}

int16_t JIT::Backend::getBranchOffset(Instructions::Instruction16 * instrStart) {
    return (instrStart - &instructions[instructionCount]) * 2; // jede Instruktion sind 16 Bit = 2 Byte
    //return &instructions[instructionCount] - instrStart;
}

JIT::Instructions::Instruction16 * JIT::Backend::getInstructions() {
    return instructions;
}

uintptr_t JIT::Backend::getThumbAddress() const {
    //return reinterpret_cast<uintptr_t>(instructions);
    return reinterpret_cast<uintptr_t>(instructions) | 0x1U;
}

void JIT::Backend::resetKernel() {
    // as no dynamic memory allocation is used, it is sufficient to just reset the instruction count/pointer
    instructionCount = 0;
}