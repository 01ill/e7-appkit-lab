#ifndef JIT_GENERATORS_GEMM_HPP
#define JIT_GENERATORS_GEMM_HPP

#include "backend/Backend.hpp"
#include "instructions/Base.hpp"
#include <cstdint>

namespace JIT {
    namespace Generators {
        class Gemm;
    }
}

class JIT::Generators::Gemm {
    private:
        Backend backend;
        /*
        If no immediates can be used, the priority has to be given to loads from B and A.
        C only has to be accessed at the first and last iteration.

        Constraints:
        - N has to be <= 255 or needs to be encoded as immediate constant (only possible in certain cases)
        - N is only used for CMP
            - if N cant be encoded as immediate, N - (N % nr) needs to be stored in register.

        - K has to be <= 511
            - if K is larger than 511, the value has to be stored in a register
        
        - M has to be <= 63
            - for CMP it needs to be <= 255 or encoded as immediate constant (only possible in certain cases)
                - if at the same time K <= 511 then the loop can be modified
                - if both is not possible, an extra register is needed just for M - (M % mr) if M isn't a multiple of mr to ensure correct microkernel execution
            - for A VLDR smaller is better, but can be done with a single 

        - provide a slow fallback which won't be needed as a 256*256*512 matrix does not fit into the storage of the Ensemble E7 board
        */
        enum RegisterImmediateStrategy : uint8_t {
            /*
            Default Strategy. can be used when
            - C can be done with immediates (M small enough <= 63) and
            - CMP can be done with immediates (N (<= 255) and M (<= 255) small enough)
            */
            ALL_IMMEDIATES = 0b0,
            USE_LDB_REGISTER = 0b1, // Default Strategy: this means use C_ROW1_Base and C_ROW2_Base if C cant use immediates
            USE_CROW1_REGISTER = 1 << 1,
            USE_CROW2_REGISTER = 1 << 2,
            USE_A_ADD_REGISTER = 1 << 3,
            USE_N_LEN_REGISTER = 1 << 4,
            USE_M_LEN_REGISTER = 1 << 5,
            USE_BCOL3_REGISTER = 1 << 6,

            /* List all possible combinations */
            // Large K and large M but priority given to using fast C loads/stores. before each CMP the value has to be loaded and afterwards removed in temp register
            USE_K_CROW1_CROW2_REGISTER = USE_LDB_REGISTER | USE_CROW1_REGISTER | USE_CROW2_REGISTER,
            USE_K_CROW1_REGISTER = USE_LDB_REGISTER | USE_CROW1_REGISTER,
            USE_K_A_CROW1_REGISTER = USE_LDB_REGISTER | USE_A_ADD_REGISTER | USE_CROW1_REGISTER,
            // Large N and large M but priority given to using fast C loads/stores. before each CMP the value has to be loaded and afterwards removed in temp register
            USE_N_CROW1_CROW2_REGISTER = USE_N_LEN_REGISTER | USE_CROW1_REGISTER | USE_CROW2_REGISTER,
            USE_N_CROW1_REGISTER = USE_N_LEN_REGISTER | USE_CROW1_REGISTER,
            // Large N and large N
            USE_K_N_LEN_REGISTER = USE_LDB_REGISTER | USE_N_LEN_REGISTER,
            USE_K_CROW1_N_LEN_REGISTER = USE_LDB_REGISTER | USE_CROW1_REGISTER | USE_N_LEN_REGISTER,
            USE_CROW1_CROW2_M_REGISTER = USE_CROW1_REGISTER | USE_CROW2_REGISTER | USE_M_LEN_REGISTER,
            // Large M
            USE_CROW1_CROW2_REGISTER = USE_CROW1_REGISTER | USE_CROW2_REGISTER,
            USE_A_CROW1_REGISTER = USE_A_ADD_REGISTER | USE_CROW1_REGISTER,
            USE_A_CROW1_CROW2_REGISTER = USE_A_ADD_REGISTER | USE_CROW1_REGISTER | USE_CROW2_REGISTER,

            // Using 4x6 Microkernel and needing to use the second K_LEN Register
            USE_K_K2_REGISTER = USE_LDB_REGISTER | USE_BCOL3_REGISTER,
            USE_K_K2_CROW1_REGISTER = USE_LDB_REGISTER | USE_BCOL3_REGISTER | USE_CROW1_REGISTER,
            USE_K_K2_N_LEN_REGISTER = USE_LDB_REGISTER | USE_BCOL3_REGISTER | USE_N_LEN_REGISTER,
            USE_K_K2_A_REGISTER = USE_LDB_REGISTER | USE_BCOL3_REGISTER | USE_A_ADD_REGISTER,
        };
        struct MicroKernelConfiguration {
            bool insertPreloadHints;
            RegisterImmediateStrategy registerStrategy;
            Instructions::Register LDB_REGISTER;
            Instructions::Register BCOL3_REGISTER;
            Instructions::Register CROW1_REGISTER;
            Instructions::Register CROW2_REGISTER;
            Instructions::Register LDC_REGISTER;
            Instructions::Register A_ADD_REGISTER;
            Instructions::Register N_LEN_REGISTER;
            Instructions::Register M_LEN_REGISTER;
        };

        void generateMicroKernel(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, MicroKernelConfiguration & configuration);
        void emitLoadB(Instructions::Register targetReg, MicroKernelConfiguration & configuration, uint32_t leftShiftAmount, uint32_t offset, bool secondHalf = false, bool try16Bit = false);
        void emitLoadStoreC(MicroKernelConfiguration & configuration, Instructions::VectorRegister targetReg, uint32_t ldc, bool store);
        void emitLoadStoreC46(Instructions::VectorRegister targetReg, uint32_t ldc, bool store = false);
    
    public:
        Gemm(Instructions::Instruction16 * globalBuffer, uint32_t bufferSize) : backend(globalBuffer, bufferSize) {}
        using Func = void (*) (float const *, float const *, float *);
        void (*generate(uint32_t m, uint32_t k, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, bool insertPreloadHints = false))(float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ c);
        // Func thumbAddressToFunc(uintptr_t thumbAddress) {
        //     __asm("dsb");
        //     __asm("isb");
        //     return reinterpret_cast<JIT::Generators::Gemm::Func>(thumbAddress);
        // }
        // Func bufferToFunc(Instructions::Instruction16 * buffer) {
        //     backend.copyToBuffer(buffer);
        //     return thumbAddressToFunc(backend.getBufferThumbAddress(buffer));
        // }
};

#endif // JIT_GENERATORS_GEMM_HPP