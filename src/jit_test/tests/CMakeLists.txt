cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

project(JitTest)

set(CMAKE_CXX_STANDARD 23)


enable_testing()
# find_package(Catch2 REQUIRED)

add_executable(jit_test 
    test_Backend.cpp
    test_DataProcessingInstructions.cpp
    test_BaseInstructions.cpp
    test_ArithmeticInstructions.cpp
    test_VectorInstructions.cpp
    catch2/catch_amalgamated.cpp
    ../instructions/DataProcessing.cpp
    ../instructions/Arithmetic.cpp
    ../instructions/Base.cpp
    ../instructions/Vector.cpp
    ../backend/Backend.cpp
    )
target_include_directories(jit_test PRIVATE ../) # add jit_test root path
target_link_libraries(jit_test)
target_compile_definitions(jit_test PRIVATE VALIDATE_ENCODINGS)

add_test(NAME jit_test COMMAND jit_test)