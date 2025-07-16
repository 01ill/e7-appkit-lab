#ifndef TIMING_HPP
#define TIMING_HPP

#include <ratio>
#ifdef M55_HP
#include "M55_HP.h"
#define CLOCK_FREQUENCY 400000000
#endif
#ifdef M55_HE
#include "M55_HE.h"
#define CLOCK_FREQUENCY 160000000
#endif

#include <chrono>
#include <cstdint>

bool RTC_Initialize();

/**
 * @brief Ruft die aktuelle Zeit vom LPRTC Modul ab.
 * Dabei wird die aktuelle Zeit aus dem RTC Register abgerufen und dann in ms umgerechnet.
 * Es wird kein Prescaler genutzt, d.h. jeder Uptick im RTC-Register passiert 32768 mal pro Sekunde.
 * D.h. 1ms ~= 3276 Ticks
 * Das Register ist 32Bit breit, d.h. es kommt nach einer Weile zum Overflow (Wrap). Dabei wird der Wert automatisch wieder auf 0 gesetzt
 * und dabei weitergezählt.
 * 2^32 / 32768 = 131072 Sekunden ~= 36 Stunden. D.h. nach anderthalb Tagen kommt es zum Overflow.
 * Es sollte also darauf geachtet werden, dass das Board nicht länger an ist.
 * Der Counter wird erst zurückgesetzt, wenn das Board ausgeschaltet wird.
 * Ein einfacher Reset setzt den Counter nicht zurück.
 * TODO: Prescaler prüfen, auf S. 548 steht, dass standardmäßig 1hz genutzt wird (bei aktiviertem Prescaler aber nur)
 * @see Hardware Reference Manual v2.8, p. 359pp
 *
 * @return Gibt die aktuelle Zeit in ms aus dem RTC Register als unsigned 32 Bit Integer zurück.
 */
uint32_t RTC_GetTimepoint();
uint32_t RTC_GetValue();
void RTC_Sleep(uint32_t ms);

bool RTC_Uninitialize();

/**
 * Implementierung einer Clock, damit die std::chrono-Bibliothek genutzt werden kann.
 *
 * https://github.com/TeensyUser/doc/wiki/implementing-a-high-resolution-teensy-clock#implementing-the-teensy-clock
 * https://eel.is/c++draft/time.clock.req
 * https://stackoverflow.com/questions/66262132/can-i-retarget-the-chrono-class-to-use-microcontroller-as-tick-generator?rq=3
*/
class RTC_Clock {
    public:    
        using rep = uint32_t;
        using period = std::milli;
        using duration = std::chrono::duration<rep, period>;
        using time_point = std::chrono::time_point<RTC_Clock, duration>;
        static constexpr bool is_steady = false;
        static time_point now() noexcept {
            return time_point{duration { RTC_GetTimepoint()}};
        }
};


void enableCpuClock();
void disableCpuClock();

class CYCCNT_Clock {
    public:
        using rep = uint32_t;
        using period = std::ratio<1, CLOCK_FREQUENCY>;
        using duration = std::chrono::duration<rep, period>;
        using time_point = std::chrono::time_point<CYCCNT_Clock, duration>;
        static constexpr bool is_steady = false;
        static time_point now() noexcept {
            time_point time = time_point{duration{ARM_PMU_Get_CCNTR()}};
            // ARM_PMU_CYCCNT_Reset();
            return time;
        }
        static void reset() noexcept {
            ARM_PMU_CYCCNT_Reset();
        }
};

#endif // TIMING_HPP