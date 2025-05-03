#include <ctime>
#include "RTE_Components.h"
#include "profiling.hpp"
#include CMSIS_device_header

extern "C" {
    void generateFpStall(float * input);
}

int main (void) {
    float hallo[4] = {0};
    setupProfilingMVEStalls();
    startCounting();
    generateFpStall(hallo);
    stopCounting();
    printCounter();

    while (1) __WFI();
}