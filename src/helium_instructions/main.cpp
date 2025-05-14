#include <ctime>
#include "RTE_Components.h"
#include "profiling.hpp"
#include CMSIS_device_header

extern "C" {
    void generateFpStall(float * input);
    void testPredication(float * input);
}

int main (void) {
    float hallo[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    setupProfilingMVEStalls();
    startCounting();
    generateFpStall(hallo);
    stopCounting();
    printCounter();

    testPredication(hallo);

    while (1) __WFI();
}