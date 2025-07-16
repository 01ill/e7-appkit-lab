#ifndef PROFILING_HPP
#define PROFILING_HPP
// https://arm-software.github.io/CMSIS_5/Core/html/group__pmu8__events__armv81.html
void setupProfilingMVEStalls();
void setupProfilingMVEInstructions();
void setupProfilingMemory();
void setupProfilingStalls();
void setupProfilingDualIssue();

void startCounting();

void stopCounting();

void printCounter();
void printCounterStalls();
void printCounterMemory();
void printCounterDualIssue();
void printCounterMVEInstructions();

#endif // PROFILING_HPP