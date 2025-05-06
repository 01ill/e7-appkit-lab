#ifndef JIT_INSTRUCTIONS_DATA_PROCESSING_HPP
#define JIT_INSTRUCTIONS_DATA_PROCESSING_HPP

#include "Base.hpp"
#include <cstdint>

namespace JIT {
    namespace Instructions {
        class DataProcessing;
    }
}

class JIT::Instructions::DataProcessing {
    public:
        /**
         * @brief 
         * 
         * @param Rt 
         * @param Rn 
         * @param imm5 
         * @return Instruction16 
         *
         * -- Performance -- (p. 29)
         * Latency: 2
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 ldrImmediate16(Register Rt, Register Rn, uint8_t imm5 = 0);
        /**
         * @brief 
         * 
         * @param Rt 
         * @param Rn 
         * @param imm12 
         * @return Instruction32 
         * @see C2.4.78 T3, p.668
         * -- Performance -- (p. 29)
         * Latency: 2
         * Throughput: 1
         */
        static Instruction32 ldrImmediate32(Register Rt, Register Rn, uint16_t imm12 = 0);
        /**
         * @brief 
         * 
         * @param Rt 
         * @param Rn 
         * @param imm8 
         * @param preIndexed 
         * @param writeBack 
         * @return Instruction32 
         * @see C2.4.78 T3, p.668
         * -- Performance -- (p. 27)
         * Latency: 2
         * Throughput: 1
         */
        static Instruction32 ldrImmediate32(Register Rt, Register Rn, uint8_t imm8 = 0, bool preIndexed = true, bool writeBack = false);

        /**
         * @brief 
         * 
         * @param Rt 
         * @param Rn 
         * @param Rm 
         * @return Instruction16 
         * @see C2.4.80, Encoding T2, p. 673
         * -- Performance -- (p. 27)
         * Latency: 2
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 ldrRegister16(Register Rt, Register Rn, Register Rm);

        /**
         * @brief Calculates address from base register value (Rn) and an offset value (Rm),
         * loads word from memory and writes it to target register (Rt).
         * Offset can be shifted left by 0-3 bits.
         * @param Rt Target Register
         * @param Rn Base Address Register
         * @param Rm Offset Register
         * @param imm2 Optional Left Shift (0-3) for the offset. Default is 0.
         * @return Instruction32 Encoded Instruction
         * @see C2.4.80, Encoding T2, p. 673
         * -- Performance -- (p. 27)
         * Latency: 2
         * Throughput: 1
        */
        static Instruction32 ldrRegister32(Register Rt, Register Rn, Register Rm, uint8_t imm2 = 0);

        static Instruction16 str(Register Rn, Register Rt);

        /**
         * MOV (Immediate): Moves an immediate to an register 
         */
        /**
         * @brief MOV Immediate: Encoding T1
         * 
         * @param Rd 
         * @param imm8 
         * @return Instruction16
         * @see C2.4.119, Encoding T1, p. 741
         * -- Performance -- (p. 25)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 movImmediate16(Register Rd, uint8_t imm8);
        
        /**
         * @brief MOV Immediate: Encoding T3
         * 
         * @param Rd 
         * @param imm16 
         * @return Instruction32 
         * @see C2.4.119, Encoding T3, p. 741
         * -- Performance -- (p. 24)
         * Latency: 1
         * Throughput: 1
         */
        static Instruction32 movImmediate32(Register Rd, uint16_t imm16);

        /**
         * MOV (Register): Copy the value of an register to the destination register 
         */
        /**
         * @brief MOV Register: Encoding T1
         * 
         * @param Rn 
         * @param Rt 
         * @return Instruction16 
         * @see C2.4.120, Encoding T1, p. 743
         * -- Performance -- (p. 25)
         * Latency: 1
         * Throughput: 2
         * Dual Issue: 11
         */
        static Instruction16 movRegister16(Register Rd, Register Rm);
        /**
         * @brief Mov Register: Encoding T2
         * 
         * @param Rd 
         * @param Rm 
         * @param shift 
         * @param amount 
         * @return Instruction16 
         * @note Sets flags (MOVS) if outside IT block
         * @see C2.4.120, Encoding T2, p. 743
         * -- Performance -- (p. 25)
         * Latency: 1
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 movRegister16(Register Rd, Register Rm, Shift shift, uint8_t amount);
        /**
         * @brief MOV Register: Encoding T3
         * 
         * @param Rd 
         * @param Rm 
         * @param shift 
         * @param amount 
         * @return Instruction32 
         * @see C2.4.120, Encoding T3, p. 744
         * -- Performance -- (p. 24)
         * Latency: 1
         * Throughput: 1
         */
        static Instruction32 movRegister32(Register Rd, Register Rm, Shift shift, uint8_t amount);
        static Instruction16 push(Register Rd);
        static Instruction16 vpush(VectorRegister Rd);
        static Instruction16 pop(Register Rd);
};

#endif // JIT_INSTRUCTIONS_DATA_PROCESSING_HPP
