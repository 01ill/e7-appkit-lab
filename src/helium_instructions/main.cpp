#include <chrono>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include "RTE_Components.h"
#include "system_M55.h"
#include CMSIS_device_header

#include "board.h"

int main (void)
{
    int32_t clock = GetSystemCoreClock();
    std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();
    int32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
    while (1) {
        clock = GetSystemCoreClock();
        time = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
        printf("SystemCoreClock: %d\n", clock);
    }
    
    int val = 1;

    while (1) __WFI();
}


// Stubs to suppress missing stdio definitions for nosys
#define TRAP_RET_ZERO  {__BKPT(0); return 0;}
int _close(int val) TRAP_RET_ZERO
int _lseek(int val0, int val1, int val2) TRAP_RET_ZERO
int _read(int val0, char * val1, int val2) TRAP_RET_ZERO
int _write(int val0, char * val1, int val2) TRAP_RET_ZERO
int _write_r(int val0, char * val1, int val2) TRAP_RET_ZERO