#ifndef BASE_HPP
#define BASE_HPP

#include <cstdint>
#ifdef PRINT_ENCODING_ERRORS
#include "SEGGER_RTT.h"
#endif

namespace JIT {
    namespace Instructions {
        class Base;
        using Instruction16 = uint16_t;
        using Instruction32 = uint32_t;
        using Reg = uint8_t;
        enum Register : Reg {
            R0 = 0,
            R1 = 1,
            R2 = 2,
            R3 = 3,
            R4 = 4,
            R5 = 5,
            R6 = 6,
            R7 = 7,
            R8 = 8,
            R9 = 9,
            R10 = 10,
            R11 = 11,
            R12 = 12,
            R13 = 13,
            R14 = 14,
            R15 = 15,
            SP = R13,
            LR = R14,
            PC = R15,
        };
        namespace LowRegisters {
            enum LowRegister : Reg {
                R0 = 0,
                R1 = 1,
                R2 = 2,
                R3 = 3,
                R4 = 4,
                R5 = 5,
                R6 = 6,
                R7 = 7,
            };    
        }
        enum FloatRegister : Reg {
            S0 = 0,
            S1 = 1,
            S2 = 2,
            S3 = 3,
            S4 = 4,
            S5 = 5,
            S6 = 6,
            S7 = 7,
            S8 = 8,
            S9 = 9,
            S10 = 10,
            S11 = 11,
            S12 = 12,
            S13 = 13,
            S14 = 14,
            S15 = 15,
            S16 = 16,
            S17 = 17,
            S18 = 18,
            S19 = 19,
            S20 = 20,
            S21 = 21,
            S22 = 22,
            S23 = 23,
            S24 = 24,
            S25 = 25,
            S26 = 26,
            S27 = 27,
            S28 = 28,
            S29 = 29,
            S30 = 30,
            S31 = 31
        };
        enum DoubleRegister : Reg {
            D0 = 0,
            D1 = 1,
            D2 = 2,
            D3 = 3,
            D4 = 4,
            D5 = 5,
            D6 = 6,
            D7 = 7,
            D8 = 8,
            D9 = 9,
            D10 = 10,
            D11 = 11,
            D12 = 12,
            D13 = 13,
            D14 = 14,
            D15 = 15
        };
        enum VectorRegister : Reg {
            Q0 = 0,
            Q1 = 1,
            Q2 = 2,
            Q3 = 3,
            Q4 = 4,
            Q5 = 5,
            Q6 = 6,
            Q7 = 7
        };
        enum Size : uint8_t {
            Size8 = 0b00,
            Size16 = 0b01,
            Size32 = 0b10,
            Size64 = 0b11
        };
        enum DataType {
            I8, I16, I32, I64, F16, F32
        };
        enum Shift : uint8_t {
            LSL = 0b00,
            LSR = 0b01,
            ASR = 0b10,
            ROR = 0b11
        };
        enum Condition : uint8_t {
            EQ = 0b0000,
            NE = 0b0001,
            CS = 0b0010,
            CC = 0b0011,
            MI = 0b0100,
            PL = 0b0101,
            VS = 0b0110,
            VC = 0b0111,
            HI = 0b1000,
            LS = 0b1001,
            GE = 0b1010,
            LT = 0b1011,
            GT = 0b1100,
            LE = 0b1101,
            AL = 0b1110
        };
    }
}

class JIT::Instructions::Base {
    public:
        template <typename... Register>
        static bool assertLowRegister(Register... regs) {
            // https://timsong-cpp.github.io/cppwp/n4868/temp.variadic#10
            return (... && (regs <= 7));
        }
        static void printValidationError(const char * message) {
            #ifdef PRINT_ENCODING_ERRORS
            SEGGER_RTT_printf(0, "%s \n", message);
            #endif
        }
        /**
         * @brief No Operation
         * 
         * @return Instruction16 
         */
        static Instruction16 nop16();
        /**
         * @brief No Operation
         * 
         * @return Instruction32 
         */
        static Instruction32 nop32();
        /*
        15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
        0  1  0  0  0  1  1  1  0  <   Rm    > 0   0  0
        
        Rm ... Register holding the address to be branched to
        */
        static Instruction16 bx(Register Rm);
        /**
            C2.4.492, T4, p. 1438
        */
        static Instruction32 dlstp(Register Rn, Size size);
        static Instruction32 dls(Register Rn);
        /**
         * @brief 
         * 
         * @param imm11 The instruction size (4 bytes) has to be added
         * @return Instruction32 
         */
        static Instruction32 letp(int16_t imm11);
        static Instruction32 le(int16_t imm1);


        /**
         * @brief Compare Immediate: Subtracts immediate value from register and updates flags. Result is discarded
         * 
         * @param Rn Source register
         * @param imm8 Immediate value
         * @return Instruction16 
         * @see C2.4.40, Encoding T1, p. 600

         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 cmpImmediate16(Register Rn, uint8_t imm8);

        /**
         * @brief Compare Immediate: Subtracts immediate value from register and updates flags. Result is discarded
         * 
         * @param Rn Source register
         * @param imm8 Immediate value
         * @return Instruction32
         * @see C2.4.40, Encoding T2, p. 600

         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 1
         */
        static Instruction32 cmpImmediate32(Register Rn, uint32_t constant);

        /**
         * @brief Compare Register
         *  Subtracts optionally-shifted (for T3) register value from register value.
         *  Updates condition flags and discards result.
         */
        /** 
         * @brief Compare Register (T1/T2)
         * @param Rn Source register
         * @param Rm Register with subtraction value
         * @return Instruction16 
         * @see C2.4.41, Encoding T1/T2, p. 602
         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 cmpRegister16(Register Rn, Register Rm);
        /**
         * @brief Compare Register (T3)
         * 
         * @param Rn Source register
         * @param Rm Register with subtraction value
         * @param shift Type of shift
         * @param amount Amount of shift
         * @return Instruction32 
         * @see C2.4.41, Encoding T3, p. 602
         * -- Performance -- (p. 19, 23) / CMP (register) (T2)
         * Latency: 1 (2 if shift is used)
         * Throughput: 1
         */
        static Instruction32 cmpRegister32(Register Rn, Register Rm, Shift shift = LSL, uint8_t amount = 0);
        /**
         * Branch with Optional Condition
         * 
         */
        /**
         * @brief 
         * 
         * @param cond 
         * @param imm8 
         * @return Instruction16 
         */
        static Instruction16 bCond16(Condition cond, int16_t imm8);
        static Instruction16 b16(int16_t imm11);
        static Instruction32 bCond32(Condition cond, int32_t label);
        static Instruction32 b32(uint32_t label);

        /**
         * @brief Generates UDF (Undefined Instruction)
         * helpful to exit the execution and print the registers
         * 
         * @return Instruction16 
         */
        static Instruction16 udf(uint8_t imm8);

        /**
         * @brief Check if instructions can encode the passed 32bit immediate
         * 
         * @return true If the instruction can encode the 32bit immediate as a constant
         * @return false If the instruction can't encode the 32bit immediate
         */
        static bool canEncodeImmediateConstant(uint32_t const);
        /**
         * @brief Encodes the immediate constant into an instruction. Only the immediate is encoded. Registers etc. are dependent on the instruction
         * and have to be encoded earlier or later.
         * Tested with CMP and AND
         * @param instr Existing instruction
         * @param constant Constant to encode
         * @return Instruction32 
         */
        static Instruction32 encodeImmediateConstant(Instruction32 instr, uint32_t constant);


        static Instruction32 pldImmediate(Register Rn, uint16_t imm, bool write = false);
};

#endif // BASE_HPP