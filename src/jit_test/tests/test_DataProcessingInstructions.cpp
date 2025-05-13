#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "catch2/catch_amalgamated.hpp"

using namespace JIT::Instructions;

TEST_CASE("LDR Immediate 16", "[LDR]") {
    SECTION("Test 1") {
        // 6835      	ldr	r5, [r6, #0]
        REQUIRE(DataProcessing::ldrImmediate16(R5, R6) == 0x6835);
        // 6811      	ldr	r1, [r2, #0]
        REQUIRE(DataProcessing::ldrImmediate16(R1, R2) == 0x6811);
    }
    SECTION("With offset") {
        // 69d1      	ldr	r1, [r2, #28]
        REQUIRE(DataProcessing::ldrImmediate16(R1, R2, 28) == 0x69d1);
    }
    SECTION("validation errors - immediate") {
        REQUIRE(DataProcessing::ldrImmediate16(R5, R6, 71) == Base::nop16());
    }
    SECTION("validation errors - high register") {
        REQUIRE(DataProcessing::ldrImmediate16(R9, R6) == Base::nop16());
        REQUIRE(DataProcessing::ldrImmediate16(R7, R10) == Base::nop16());
    }
}

TEST_CASE("LDR Immediate 32", "[LDR]") {
    SECTION("Test 1 - Encoding T3") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2) == 0xf8d2'9000);
    }
    SECTION("Test 2 - Encoding T3") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, 0xfe5) == 0xf8d2'9fe5);
    }
    SECTION("validation errors - immediate (T3)") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, 0x1fff) == Base::nop32());
    }
    SECTION("Test 1 - Encoding T4") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, -123) == 0xf852'9c7b);
    }
    SECTION("Test 2 - Encoding T4") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, 251, true, true) == 0xf852'9ffb);
    }
    SECTION("Test 3 - Encoding T4") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, 251, false, false) == 0xf852'9bfb);
    }
    SECTION("Test 4 validation - Encoding T4") {
        REQUIRE(DataProcessing::ldrImmediate32(R9, R2, 256, false, true) == Base::nop32());
    }
}

TEST_CASE("LDR (register) 16 encodes correctly", "[LDR]") {
    SECTION("Test 1") {
        // 59f5      	ldr	r5, [r6, r7]
        REQUIRE(DataProcessing::ldrRegister16(R5, R6, R7) == 0x59f5);
    }
    SECTION("validation errors") {
        REQUIRE(DataProcessing::ldrRegister16(R5, R6, R8) == Base::nop16());
        REQUIRE(DataProcessing::ldrRegister16(R8, R6, R5) == Base::nop16());
        REQUIRE(DataProcessing::ldrRegister16(R5, R9, R4) == Base::nop16());
    }
}

TEST_CASE("LDR (register) 32 Bit encodes correctly", "[LDR]") {
    SECTION("LDR: no shift") {
        //  f852 a008 	ldr.w	sl, [r2, r8]
        REQUIRE(DataProcessing::ldrRegister32(R10, R2, R8) == 0xf852'a008);
    }
    SECTION("LDR: left shift") {
        //  f851 9023 	ldr.w	r9, [r1, r3, lsl #2]
        REQUIRE(DataProcessing::ldrRegister32(R9, R1, R3, 2) == 0xf851'9023);
    }
    SECTION("validation errors") {
        REQUIRE(DataProcessing::ldrRegister32(R10, R9, R8, 4) == Base::nop32());
        REQUIRE(DataProcessing::ldrRegister32(R10, R9, SP, 3) == Base::nop32());
        REQUIRE(DataProcessing::ldrRegister32(R10, R9, PC, 3) == Base::nop32());
    }
}

TEST_CASE("MOV Immediate 16 Bit encodes correctly", "[MOV]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::movImmediate16(R5, 231) == 0x25e7);
    }
    SECTION("validation errors") {
        REQUIRE(DataProcessing::movImmediate16(R9, 0) == Base::nop16());
    }
}

TEST_CASE("MOV Immediate 32 Bit encodes correctly", "[MOV]") {
    SECTION("Test 1") {
       REQUIRE(DataProcessing::movImmediate32(R5, 231) == 0xf240'05e7);
    }
    SECTION("Test 2") {
        REQUIRE(DataProcessing::movImmediate32(R5, 0xffff) == 0xf64f'75ff); 
    }
}

TEST_CASE("MOV Register 16 Bit encodes correctly", "[MOV]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::movRegister16(R10, R11) == 0x46da);
    }
    SECTION("Test 2") {
        REQUIRE(DataProcessing::movRegister16(R4, R3, LSL, 4) == 0x011c);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::movRegister16(R8, R7, LSL, 1) == Base::nop16());
        REQUIRE(DataProcessing::movRegister16(R7, R9, LSL, 1) == Base::nop16());
        REQUIRE(DataProcessing::movRegister16(R7, R6, LSL, 32) == Base::nop16());
        REQUIRE(DataProcessing::movRegister16(R7, R6, ROR) == Base::nop16());
    }
}

TEST_CASE("MOV Register 32 Bit encodes correctly", "[MOV]") {
    SECTION("Test 1 - RRX") {
        REQUIRE(DataProcessing::movRegister32(R9, R10, ROR) == 0xea4f'093a);
        REQUIRE(DataProcessing::movRegister32(R9, R10, ROR, 0, true) == 0xea5f'093a);
    }
    SECTION("Test - other shifts") {
        REQUIRE(DataProcessing::movRegister32(R9, R10, LSL, 4, true) == 0xea5f'190a);
        REQUIRE(DataProcessing::movRegister32(R9, R10, LSL, 29, true) == 0xea5f'794a);
        REQUIRE(DataProcessing::movRegister32(R9, R10, LSL, 29, false) == 0xea4f'794a);
        REQUIRE(DataProcessing::movRegister32(R9, R10, ASR, 29, false) == 0xea4f'796a);
        REQUIRE(DataProcessing::movRegister32(R9, R10, LSR, 29, false) == 0xea4f'795a);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::movRegister32(R9, R10, ROR, 1, true) == Base::nop32());
    }
}

TEST_CASE("PUSH 16 Bit encodes correctly", "[PUSH]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::push16(R1, R2, R3) == 0xb40e);
        REQUIRE(DataProcessing::push16(R1, R2, R3, R7, LR) == 0xb58e);
        REQUIRE(DataProcessing::push16(R0) == 0xb401);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::push16(R8) == Base::nop16());
    }
}

TEST_CASE("PUSH 32 Bit encodes correctly", "[PUSH]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::push32(R1, R2, R3, R7, LR) == 0xe92d'408e);
        REQUIRE(DataProcessing::push32(R1, R2, R12, R7, LR) == 0xe92d'5086);
        REQUIRE(DataProcessing::push32(R10, R11) == 0xe92d'0c00);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::push32(SP) == Base::nop32());
        REQUIRE(DataProcessing::push32(R1, PC) == Base::nop32());
    }
}

TEST_CASE("POP 16 Bit encodes correctly", "[POP]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::pop16(R3, R7, PC) == 0xbd88);
        REQUIRE(DataProcessing::pop16(PC) == 0xbd00);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::pop16(R3, R8) == Base::nop16());
        REQUIRE(DataProcessing::pop16(LR) == Base::nop16());
    }
}

TEST_CASE("POP 32 Bit encodes correctly", "[POP]") {
    SECTION("Test 1") {
        REQUIRE(DataProcessing::pop32(R10, PC) == 0xe8bd'8400);
        REQUIRE(DataProcessing::pop32(R1, R7, R8, LR) == 0xe8bd'4182);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::pop32(R10, LR, PC) == Base::nop32());
        REQUIRE(DataProcessing::pop32(SP) == Base::nop32());
    }
}

TEST_CASE("VPUSH encodes correctly", "[VPUSH]") {
    SECTION("Test 1 - Push QRegisters") {
        REQUIRE(DataProcessing::vpush(Q4) == 0xed2d'8b04);
        REQUIRE(DataProcessing::vpush(Q6, 2) == 0xed2d'cb08);
        REQUIRE(DataProcessing::vpush(Q0, 8) == 0xed2d'0b20);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::vpush(Q6, 3) == Base::nop32());
    }
    SECTION("Test 2 - Push DRegisters") {
        REQUIRE(DataProcessing::vpush(D3, 3) == 0xed2d'3b06);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::vpush(D3, 14) == Base::nop32());
    }
}

TEST_CASE("VPOP encodes correctly", "[VPOP]") {
    SECTION("Test 1 - POP QRegisters") {
        REQUIRE(DataProcessing::vpop(Q4) == 0xecbd'8b04);
        REQUIRE(DataProcessing::vpop(Q6, 2) == 0xecbd'cb08);
        REQUIRE(DataProcessing::vpop(Q0, 8) == 0xecbd'0b20);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::vpop(Q6, 3) == Base::nop32());
    }
    SECTION("Test 2 - POP DRegisters") {
        REQUIRE(DataProcessing::vpop(D3, 3) == 0xecbd'3b06);
    }
    SECTION("validate errors") {
        REQUIRE(DataProcessing::vpop(D3, 14) == Base::nop32());
    }
}
