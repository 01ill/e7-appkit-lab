#ifndef PROFILING_HPP
#define PROFILING_HPP
// https://arm-software.github.io/CMSIS_5/Core/html/group__pmu8__events__armv81.html
void setupProfilingMVEStalls();
void setupProfilingMVEInstructions();
void setupProfilingMemory();
void setupProfilingStalls();

void startCounting();

void stopCounting();

void printCounter();
void printCounterStalls();

#endif // PROFILING_HPP