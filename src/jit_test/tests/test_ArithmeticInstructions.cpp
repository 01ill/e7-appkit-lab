#include "catch2/catch_amalgamated.hpp"
#include "instructions/Base.hpp"
#include "instructions/DataProcessing.hpp"
#include "instructions/Arithmetic.hpp"
#include "instructions/Vector.hpp"
#include <cstdio>

using namespace JIT::Instructions;

TEST_CASE("ADD Immediate (T1) encodes correctly", "[ADD]") {
    SECTION("all low") {
        REQUIRE(Arithmetic::addImmediate16(JIT::Instructions::R1, R0, 5) == 0x1d41);
    }
}

TEST_CASE("ADD Immediate (T2) encodes correctly", "[ADD]") {
    SECTION("all low") {
        REQUIRE(Arithmetic::addImmediate16(JIT::Instructions::R5, 123) == 0x357b);
    }
}

TEST_CASE("ADD Immediate (T3) encodes correctly", "[ADD]") {
    SECTION("test 1") {
        REQUIRE(Arithmetic::addImmediate32(R3, R10, 32) == 0xf20a'0320);
    }
    SECTION("test 2") {
        REQUIRE(Arithmetic::addImmediate32(R7, R11, 1023) == 0xf20b'37ff);
    }
    SECTION("test 3") {
        REQUIRE(Arithmetic::addImmediate32(R7, R11, 2023) == 0xf20b'77e7);
    }
    SECTION("test 4") {
        REQUIRE(Arithmetic::addImmediate32(R7, R11, 4095) == 0xf60b'77ff);
    }
}

TEST_CASE("ADD Register (T1) encodes correctly", "[ADD]") {
    SECTION("test 1") {
        REQUIRE(Arithmetic::addRegister16(R3, R5, R7) == 0x19eb);
    }
}

TEST_CASE("ADD Register (T2) encodes correctly", "[ADD]") {
    SECTION("test 1") {
        REQUIRE(Arithmetic::addRegister16(R3, R10) == 0x4453);
    }
    SECTION("test 2") {
        REQUIRE(Arithmetic::addRegister16(R11, R9) == 0x44cb);
    }
}

TEST_CASE("ADD Register (T3) encodes correctly", "[ADD]") {
    SECTION("test 1") {
        REQUIRE(Arithmetic::addRegister32(R1, R11, R6, LSL, 2) == 0xeb0b'0186);
    }
    SECTION("test 2 (Rd = Rn)") {
        REQUIRE(Arithmetic::addRegister32(R11, R11, R3, LSL, 2) == 0xeb0b'0b83);
    }
}
