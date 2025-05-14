#include "instructions/Base.hpp"
#include "instructions/Vector.hpp"
#include "catch2/catch_amalgamated.hpp"

using namespace JIT::Instructions;

TEST_CASE("VMOV Immediate encodes correctly", "[VMOV]") {
    SECTION("VMOV Int8") {
        REQUIRE(Vector::vmovImmediate(Q7, 123, DataType::I8) == 0xef87'ee5b);
        REQUIRE(Vector::vmovImmediate(Q1, 255, DataType::I8) == 0xff87'2e5f);
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::I8) == 0xef80'2e50);
    }
    SECTION("VMOV Int16") {
        REQUIRE(Vector::vmovImmediate(Q7, 123, DataType::I16) == 0xef87'e85b);
        REQUIRE(Vector::vmovImmediate(Q1, 255, DataType::I16) == 0xff87'285f);
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::I16) == 0xef80'2850);
    }
    SECTION("VMOV Int32") {
        REQUIRE(Vector::vmovImmediate(Q7, 123, DataType::I32) == 0xef87'e05b);
        REQUIRE(Vector::vmovImmediate(Q1, 255, DataType::I32) == 0xff87'205f);
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::I32) == 0xef80'2050);
    }
    SECTION("unsupported datatypes") {
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::I64) == Base::nop32());
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::F32) == Base::nop32());
        REQUIRE(Vector::vmovImmediate(Q1, 0, DataType::F16) == Base::nop32());
    }
}

TEST_CASE("VORR encodes correctly", "[VORR]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vorr(Q1, Q2, Q3) == 0xef24'2156);
        REQUIRE(Vector::vorr(Q7, Q5, Q3) == 0xef2a'e156);
    }
}

TEST_CASE("VMOV Register encodes correctly", "[VMOV]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vmovRegister(Q7, Q6) == 0xef2c'e15c);
        REQUIRE(Vector::vmovRegister(Q3, Q0) == 0xef20'6150);
    }
}

TEST_CASE("VMOV GP<->Scalar encodes correctly", "[VMOV]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vmovGPxScalar(true, S12, R11) == 0xee16'ba10);
        REQUIRE(Vector::vmovGPxScalar(false, S12, R11) == 0xee06'ba10);
    }
}

TEST_CASE("VFMA encodes correctly", "[VFMA]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vfma(Q3, Q5, Q7) == 0xef0a'6c5e);
        REQUIRE(Vector::vfma(Q1, Q4, Q0, true) == 0xef18'2c50);
    }
}

TEST_CASE("VFMA Vector*Scalar encodes correctly", "[VFMA]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vfmaVectorByScalarPlusVector(Q7, Q0, R11) == 0xee31'ee4b);
        REQUIRE(Vector::vfmaVectorByScalarPlusVector(Q3, Q1, R2, true) == 0xfe33'6e42);
    }
}

TEST_CASE("VLDRW encodes correctly", "[VLDR]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vldrw(Q3, R11, 4) == 0xed9b'7f01);
        REQUIRE(Vector::vldrw(Q3, R11, 508) == 0xed9b'7f7f);
        REQUIRE(Vector::vldrw(Q3, R11, -4) == 0xed1b'7f01);
        REQUIRE(Vector::vldrw(Q3, R11, -508) == 0xed1b'7f7f);
    }

    SECTION("validate errors") {
        REQUIRE(Vector::vldrw(JIT::Instructions::Q3, R11, 3) == Base::nop32());
        REQUIRE(Vector::vldrw(Q3, R11, 512) == Base::nop32());
    }
}

TEST_CASE("VSTRW encodes correctly", "[VSTRW]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vstrw(Q3, R11, 4) == 0xed8b'7f01);
        REQUIRE(Vector::vstrw(Q3, R11, 508) == 0xed8b'7f7f);
        REQUIRE(Vector::vstrw(Q3, R11, -4) == 0xed0b'7f01);
        REQUIRE(Vector::vstrw(Q3, R11, -508) == 0xed0b'7f7f);
    }

    SECTION("validate errors") {
        REQUIRE(Vector::vstrw(JIT::Instructions::Q3, R11, 3) == Base::nop32());
        REQUIRE(Vector::vstrw(Q3, R11, 512) == Base::nop32());
    }
}

TEST_CASE("VCTP encodes correctly", "[VCTP]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vctp(Size32, R3) == 0xf023'e801);
    }
}

TEST_CASE("VPST encodes correctly", "[VPST]") {
    SECTION("Test 1") {
        REQUIRE(Vector::vpst(1) == 0xfe71'0f4d);
        REQUIRE(Vector::vpst(2) == 0xfe31'8f4d);
        REQUIRE(Vector::vpst(3) == 0xfe31'4f4d);
        REQUIRE(Vector::vpst(4) == 0xfe31'2f4d);
    }

    SECTION("validate errors") {
        REQUIRE(Vector::vpst(5) == Base::nop32());
        REQUIRE(Vector::vpst(0) == Base::nop32());
    }
}