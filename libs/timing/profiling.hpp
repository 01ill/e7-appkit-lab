#ifndef PROFILING_HPP
#define PROFILING_HPP
// https://arm-software.github.io/CMSIS_5/Core/html/group__pmu8__events__armv81.html
void setupProfilingMVEStalls();
void setupProfilingMVEInstructions();
void setupProfilingMemory();

void startCounting();

void stopCounting();

void printCounter();

#endif // PROFILING_HPP