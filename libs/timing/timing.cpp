#include <cstdint>

#include "timing.hpp"
#include "LPRTC.hpp"


/**
 * Vorgehen: Initialisierung -> PowerOn -> Prescaler deaktiviern -> Counter abfragen x-mal -> PowerOff -> Uninitialize
*/
bool RTC_Initialize() {
    LPRTC::getInstance().enable();
    /*int32_t ret = RTCdrv->Initialize(RTC_Event);
    if (ret != ARM_DRIVER_OK) {
        return false;
    }
    ret = RTCdrv->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        RTC_Uninitialize();
        return false;
    }
    ret = disablePrescaler();
    if (ret != ARM_DRIVER_OK) {
        RTC_Uninitialize();
        return false;
    }*/

    return true;
}

bool RTC_Uninitialize() {
    LPRTC::getInstance().disable();
    return true;
}

uint32_t RTC_GetTimepoint() {
    uint32_t val = LPRTC::getInstance().getCurrentValue();
    // Umwandeln von 32768Hz in ms
    return (int)(val / 32.768f);
}

uint32_t RTC_GetValue() {
    return LPRTC::getInstance().getCurrentValue();
}

void RTC_Sleep(uint32_t ms) {
    uint32_t start = RTC_GetTimepoint();
    while (RTC_GetTimepoint() - start < ms) { }
}