#define CATCH_CONFIG_MAIN


#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "catch2/catch_amalgamated.hpp"
#include "instructions/Vector.hpp"
#include <cstdio>

using namespace JIT::Instructions;

TEST_CASE("ldr encodes correctly", "[LDR]") {
    SECTION("ldr: both low register") {
        REQUIRE(JIT::Instructions::DataProcessing::ldr(JIT::Instructions::R0, JIT::Instructions::R1) == 0x6801);
    }
}

TEST_CASE("LDR (register) 32 Bit encodes correctly", "[LDR]") {
    SECTION("LDR: no shift") {
        //  f852 a008 	ldr.w	sl, [r2, r8]
        REQUIRE(DataProcessing::ldrRegister(R10, R2, R8) == 0xf852'a008);
    }
    SECTION("LDR: left shift") {
        //  f851 9023 	ldr.w	r9, [r1, r3, lsl #2]
        REQUIRE(DataProcessing::ldrRegister(R9, R1, R3, 2) == 0xf851'9023);
    }
}