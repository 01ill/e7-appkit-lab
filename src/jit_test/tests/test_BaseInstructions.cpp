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

TEST_CASE("CMP Immediate 32 encodes correctly", "[CMP]") {
    SECTION("CMP Immediate 32") {
        REQUIRE(Base::cmpImmediate32(R10, 0x00af00af) == 0xf1ba1faf);
        REQUIRE(Base::cmpImmediate32(R10, 0xafafafaf) == 0xf1ba3faf);
        REQUIRE(Base::cmpImmediate32(R10, 0xaf00af00) == 0xf1ba2faf);
        REQUIRE(Base::cmpImmediate32(R10, 0x8f00'0000) == 0xf1ba'4f0f);
        REQUIRE(Base::cmpImmediate32(R10, 0x0000'01fe) == 0xf5ba'7fff);
    }

    SECTION("invalid immediates") {
        REQUIRE(Base::cmpImmediate32(R10, 0x0000'01ff) == Base::nop32());
        REQUIRE(Base::cmpImmediate32(R10, 0x001f'00fe) == Base::nop32());
        REQUIRE(Base::cmpImmediate32(R10, 0x1000'0001) == Base::nop32());
        REQUIRE(Base::cmpImmediate32(R10, 0x0110'1000) == Base::nop32());
    }
}
/*
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
}*/


TEST_CASE("Immediate Constants", "[Immediate]") {
    SECTION("Correct Immediates") {
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'00ae) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'00fe) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'0001) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'0000) == true);
        
        REQUIRE(Base::canEncodeImmediateConstant(0x00ae'00ae) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x00fe'00fe) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0001'0001) == true);

        REQUIRE(Base::canEncodeImmediateConstant(0xae00'ae00) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0xfe00'fe00) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0100'0100) == true);


        REQUIRE(Base::canEncodeImmediateConstant(0xaeae'aeae) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0xfefe'fefe) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0101'0101) == true);


        REQUIRE(Base::canEncodeImmediateConstant(0xaf00'0000) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x4f80'0000) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x2fc0'0000) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'07c8) == true);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'01fe) == true);
    }

    SECTION("Incorrect Immediates") {
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'01ff) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0x001f'00fe) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0x1000'0001) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0x0110'1000) == false);
        
        REQUIRE(Base::canEncodeImmediateConstant(0x2fc0'0100) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'17c8) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0x0000'11fe) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0xfeae'aeae) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0xaefe'aeae) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0xaeae'afae) == false);
        REQUIRE(Base::canEncodeImmediateConstant(0xaeae'afaf) == false);
    }
}