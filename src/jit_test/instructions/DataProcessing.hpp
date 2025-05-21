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
         * @brief Load Value from Register Adress and adds immediate
         * 
         * @param Rt Target Register
         * @param Rn Register with Adress
         * @param imm5 Offset
         * @return Instruction16 
         * @see C2.4.78 T1, p. 667
         * -- Performance -- (p. 29)
         * Latency: 2
         * Throughput: 1
         * Dual Issue: 01
         */
        static Instruction16 ldrImmediate16(Register Rt, Register Rn, uint8_t imm5 = 0);
        /**
         * @brief Loads value of address stored in Register to target register.
         * Uses Encoding T3 if immediate is positive, Pre-Index is used an no write back is used.
         * Else Encoding T4 is used.
         * @param Rt Target Register
         * @param Rn Base Address Register
         * @param imm imm8/imm12 Offset.
         * @param preIndexed Increment before accessing the value
         * @param writeBack If value should be written back. Must be set if using postIndex
         * @return Instruction32 
         * @see C2.4.78 T3/T4, p.668
         * -- Performance -- (p. 27 (T4), p. 29 (T3))
         * Latency: 2
         * Throughput: 1
         */
        static Instruction32 ldrImmediate32(Register Rt, Register Rn, int16_t imm = 0, bool preIndexed = true, bool writeBack = false);

        /**
         * @brief Loads value from address stored in register with offset stored in other register to target register
         * 
         * @param Rt Target register
         * @param Rn Base-address register
         * @param Rm Offset register
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
        static Instruction32 movtImmediate32(Register Rd, uint16_t imm16);

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
        // static Instruction16 movRegister16(Register Rd, Register Rm);
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
        static Instruction16 movRegister16(Register Rd, Register Rm, Shift shift = LSL, uint8_t amount = 0);
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
        static Instruction32 movRegister32(Register Rd, Register Rm, Shift shift = LSL, uint8_t amount = 0, bool setFlags = false);

        /**
         * @brief PUSH Encoding T2
         * 
         * @tparam Regs 
         * @param regs Register to be pushed to the stack. Lowest registers have the lowest address
         * @return Instruction16 
         * -- Performance -- (p. 30)
         * N=floor((num_regs+1)/2)
         * Latency: N+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/N (e.g. for one register: 1, four register: 1/2)
         * Dual-Issue: 00
         */
        template<typename... Regs>
        static Instruction16 push16(Regs... regs) {
            if (!Base::assertLowRegister(regs...) && ((regs != LR) && ...)) {
                Base::printValidationError("push16: only low registers and LR allowed - returning nop");
                return Base::nop16();
            }
            Instruction16 instr = 0xb400;
            // for each reg: instr |= 1 << reg using fold operation
            ((instr |= (regs == LR ? 1U << 8U : 1U << regs)), ...);
            return instr;
        }
        
        /**
         * @brief PUSH Encoding T1
         * 
         * @tparam Regs 
         * @param regs Register to be pushed to the stack. Lowest registers have the lowest address
         * @return Instruction32
         * -- Performance -- (p. 30)
         * N=floor((num_regs+1)/2)
         * Latency: N+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/N (e.g. for one register: 1, four register: 1/2)
         */
        template<typename... Regs>
        static Instruction32 push32(Regs... regs) {
            if (((regs == SP || regs == PC) || ...)) {
                Base::printValidationError("push32: SP and PC not allowed - returning nop");
                return Base::nop32();
            }
            Instruction32 instr = 0xe92d'0000;
            ((instr |= 1U << regs), ...);
            return instr;
        }

        /**
         * @brief POP Encoding T3
         * 
         * @tparam Regs 
         * @param regs Register to be popped from the stack. Lowest registers have the lowest address
         * @return Instruction16 
         * @see C2.4.144 T3, p. 794
         * -- Performance -- (p. 29)
         * N=floor((num_regs+1)/2)
         * Latency: N+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/N (e.g. for one register: 1, four register: 1/2)
         * Dual-Issue: 00
         */
        template <typename... Regs>
        static Instruction16 pop16(Regs... regs) {
            if (!Base::assertLowRegister(regs...) && ((regs != PC) && ...)) {
                Base::printValidationError("pop16: only low registers and PC allowed - returning nop");
                return Base::nop16();
            }
            Instruction16 instr = 0xbc00;
            ((instr |= (regs == PC ? 1 << 8 : 1 << regs)), ...);
            return instr;
        }

        /**
         * @brief POP Encoding T2
         * 
         * @tparam Regs 
         * @param regs Register to be pushed to the stack. Lowest registers have the lowest address
         * @return Instruction32 
         * @see C2.4.144 T2, p. 794
         * -- Performance -- (p. 28) (LDMIA T2)
         * N=floor((num_regs+1)/2)
         * Latency: N+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/N (e.g. for one register: 1, four register: 1/2)
         */
        template <typename... Regs>
        static Instruction32 pop32(Regs... regs) {
            // only one of PC or LR is allowed and SP not allowed
            if (((regs == SP) || ...) || (((regs == LR) || ... ) && ((regs == PC) || ...))) {

                Base::printValidationError("pop32: SP not allowed and only one of LR or PC allowed - returning nop");
                return Base::nop32();
            }
            Instruction32 instr = 0xe8bd'0000;
            ((instr |= 1U << regs), ...);
            return instr;
        }
        
        /**
         * @brief PUSH Double Registers to the stack. The registers need to be continuous
         * 
         * @param startRegister First register to be pushed
         * @param registerCount count of registers to be pushed
         * @return Instruction32 
         * @see C2.4.432 T1, p. 1304
         * -- Performance -- (p. 33) (VSTM T1)
         * Latency: (num_regs/2)+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/((num_regs/2)+1) (e.g. for one register: 1/2, four register: 1/3)
         */
        static Instruction32 vpush(DoubleRegister startRegister, uint8_t registerCount = 1);
        /**
         * @brief Uses the vpush function to push vector registers to the stack.
         * For each vector register two double registers are pushed to the stack
         * 
         * @param startRegister First register to be pushed
         * @param registerCount count of registers to be pushed
         * @return Instruction32 
         */
        static Instruction32 vpush(VectorRegister startRegister, uint8_t registerCount = 1);

        /**
         * @brief POP Double Registers from the stack. The registers need to be continuous
         * 
         * @param startRegister First register to be popped
         * @param registerCount count of registers to be popped
         * @return Instruction32 
         * @see C2.4.427 T1, p. 1291
         * -- Performance -- (p. 32) (VLDM T1)
         * Latency: (num_regs/2)+1 (e.g. for one register: 2, four registers: 3)
         * Throughput: 1/((num_regs/2)+1) (e.g. for one register: 1/2, four register: 1/3)
         */
        static Instruction32 vpop(DoubleRegister startRegister, uint8_t registerCount = 1);
        /**
         * @brief Uses the vpop function to pop vector registers from the stack.
         * For each vector register two double registers are popped from the stack
         * 
         * @param startRegister First register to be popped
         * @param registerCount count of registers to be popped
         * @return Instruction32 
         */
        static Instruction32 vpop(VectorRegister startRegister, uint8_t registerCount = 1);

        // static Instruction16 vpush(VectorRegister Rd);
};

#endif // JIT_INSTRUCTIONS_DATA_PROCESSING_HPP
