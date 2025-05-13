#define CATCH_CONFIG_MAIN

#include "catch2/catch_amalgamated.hpp"
#include "instructions/Base.hpp"

using namespace JIT::Instructions;

TEST_CASE("NOP encodes correctly", "[NOP]") {
    SECTION("NOP Encoding T1") {
        REQUIRE(Base::nop16() == 0xbf00);
    }

    SECTION("NOP Encoding T2") {
        REQUIRE(Base::nop32() == 0xf3af'8000);
    }
}

TEST_CASE("DLSTP encodes correctly", "[DLSTP]") {
    SECTION("Test 1") {
        REQUIRE(Base::dlstp(R5, Size32) == 0xf025'e001);
    }

    SECTION("Test 2") {
        REQUIRE(Base::dlstp(R11, Size64) == 0xf03b'e001);
    }

    SECTION("Test 3") {
        REQUIRE(Base::dlstp(R1, Size8) == 0xf001'e001);
    }

    SECTION("Test 4") {
        REQUIRE(Base::dlstp(R6, Size16) == 0xf016'e001);
    }
}

TEST_CASE("LETP encodes correctly", "[LETP]") {
    SECTION("Test 1") {
        REQUIRE(Base::letp(-8) == 0xf01f'c005);
    }
}

TEST_CASE("CMP encodes correctly", "[CMP]") {
    SECTION("CMP Immediate 16 - 1") {
        REQUIRE(Base::cmpImmediate16(R3, 9) == 0x2b09);
    }
    SECTION("CMP Immediate 16 - 2") {
        REQUIRE(Base::cmpImmediate16(R3, 231) == 0x2be7);
    }
    SECTION("CMP Register 16 - 1") {
        REQUIRE(Base::cmpRegister16(R1, R2) == 0x4291);
    }
    SECTION("CMP Register 16 - 2") {
        REQUIRE(Base::cmpRegister16(R9, R10) == 0x45d1);
    }
    SECTION("CMP Register 16 - 3") {
        REQUIRE(Base::cmpRegister16(R3, R10) == 0x4553);
    }
}

TEST_CASE("B encodes correctly", "[B]") {
    SECTION("Bcond16 - 1") {
        REQUIRE(Base::bCond16(GE, 6) == 0xda01);
    }
    SECTION("Bcond16 - 2") {
        REQUIRE(Base::bCond16(GE, 208) == 0xda66);
    }
    SECTION("Bcond16 - 3") {
        REQUIRE(Base::bCond16(GE, 2) == 0xdaff);
    }
}