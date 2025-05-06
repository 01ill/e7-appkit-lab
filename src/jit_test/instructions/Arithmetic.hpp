#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include "Base.hpp"
#include <cstdint>

namespace JIT {
    namespace Instructions {
        class Arithmetic;
    }
}

class JIT::Instructions::Arithmetic {
    public:
        /**
         * Add Immediate. Adds Immediate value to a register (Rn) 
         * and writes result to the destination register (Rd)
         */
        
        /**
         * @brief Add Immediate: Encoding T1
         * sets flags if not in an IT block
         * @param Rd Destination Register
         * @param Rn Source Register
         * @param imm3 Immediate which is added
         * @return Instruction16 Encoded Instruction
         * @see C2.4.5 T1, p. 537
         * 
         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 addImmediate16(Register Rd, Register Rn, uint8_t imm3);

        /** 
         * @brief Add Immediate: Encoding T2 
         * sets flags if not in an IT block
         * @param Rdn Intermediate and destination register
         * @param imm8 8-Bit immediate which is added to Rn
         * @return Instruction16 Encoded instruction
         * @see C2.4.5 T2, p. 537
         *
         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 addImmediate16(Register Rdn, uint8_t imm8);

        /**
         * @brief Add Immediate: Encoding T3/T4
         * 
         * @param Rd Destination Register
         * @param Rn Source Register
         * @param imm12 12-Bit unsigned immediate (between 0-4095)
         * @return Instruction32 
         * @see C2.4.5 T3/T4, p. 537
         *
         * -- Performance -- (p. 19)
         * Latency: 1/1
         * Throughput: 1/1
         */
        static Instruction32 addImmediate32(Register Rd, Register Rn, uint16_t imm12, bool setFlags = false);

        /**
         * Add Register. Adds register value and optionally shifted register value together and writes to destination register
         */
        
        /**
         * @brief Add Register: Encoding T1
         * 
         * @param Rd Destination Register
         * @param Rn Source Register 1
         * @param Rm Source Register 2
         * @return Instruction16 
         * @see C2.4.7 T1, p. 542
         *
         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 addRegister16(Register Rd, Register Rn, Register Rm);
        /**
         * @brief Add Register: Encoding T2
         * 
         * @param Rd Destination + Source Register 1
         * @param Rm Source Register 2
         * @return Instruction16 
         * @see C2.4.7 T2, p. 542
         *
         * -- Performance -- (p. 23)
         * Latency: 1
         * Throughput: 1
         * Dual Issue: 01 (00 when Rd/Rm=PC)
         */
        static Instruction16 addRegister16(Register Rdn, Register Rm);
        /**
         * @brief Add Register: Encoding T3
         * 
         * @param Rd Destination register
         * @param Rn Source register 1
         * @param Rm Source register 2
         * @param shift Type of shift
         * @param amount Amount of shift
         * @return Instruction32 
         * @see C2.4.7 T3, p. 542
         *
         * -- Performance -- (p. 19, 23)
         * Latency: 1 (2 if shift is used)
         * Throughput: 1
         */
        static Instruction32 addRegister32(Register Rd, Register Rn, Register Rm, Shift shift = LSL, uint8_t amount = 0, bool setFlags = false);
        static Instruction32 addRegister32(Register Rd, Register Rm, Shift shift = LSL, uint8_t amount = 0, bool setFlags = false);
        /**
         * SUB (Immediate): Subtracts immediate value from register and writes result to destination register
         */
        /**
         * @brief SUB (Immediate): Encoding T1
         * 
         * @param Rd Destination Register
         * @param Rn Source Register
         * @param imm3 Immediate Value to subtract
         * @return Instruction16 
         * @note Sets flags (SUBS) if outside IT block
         * @see C2.4.239, Encoding T1, p. 941
         *
         * -- Performance -- (p. 24)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 subImmediate16(Register Rd, Register Rn, uint8_t imm3);
        /**
         * @brief SUB (Immediate): Encoding T2
         * 
         * @param Rdn Destination and Source Register
         * @param imm8 Immediate Value to subtract
         * @return Instruction16 
         * @note Sets flags (SUBS) if outside IT block
         * @see C2.4.239, Encoding T2, p. 941
         *
         * -- Performance -- (p. 24)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 subImmediate16(Register Rdn, uint8_t imm8);

        /**
         * @brief SUB (Immediate): Encoding T4
         * 
         * @param Rd Destination Register
         * @param Rn Source Register
         * @param imm12 Immediate Value to subtract
         * @return Instruction32 
         * @see C2.4.239, Encoding T4, p. 942
         *
         * -- Performance -- (p. 19), SUBW (immediate) (T4)
         * Latency: 1
         * Throughput: 1
         */
        static Instruction32 subImmediate32(Register Rd, Register Rn, uint16_t imm12);

        /**
         * SUB (register): Subtracts register value (Rm) from other register value (Rn)
         * and writes the result to destination register (Rd)
         */
        /**
         * @brief 
         * 
         * @param Rd 
         * @param Rn 
         * @param Rm 
         * @return Instruction16 
         * @see C2.4.241, Encoding T1, p. 945
         *
         * -- Performance -- (p. 24)
         * Latency: 1
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 subRegister16(Register Rd, Register Rn, Register Rm);

        /**
         * @brief 
         * 
         * @param Rd 
         * @param Rn 
         * @param Rm 
         * @param shift 
         * @param amount 
         * @param setFlags 
         * @return Instruction32 
         * @see C2.4.241, Encoding T1, p. 945
         *
         * -- Performance -- (p. 19, 23)
         * Latency: 1 (2 if shift is used)
         * Throughput: 1
         */
        static Instruction32 subRegister32(Register Rd, Register Rn, Register Rm, Shift shift = LSL, uint8_t amount = 0, bool setFlags = false);
        static Instruction32 subRegister32(Register Rd, Register Rm, Shift shift = LSL, uint8_t amount = 0, bool setFlags = false);
};

#endif // ARITHMETIC_HPP