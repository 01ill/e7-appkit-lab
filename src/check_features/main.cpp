#include "RTE_Components.h"
#include CMSIS_device_header // needed for WFI/WFE
#include <cstdint>
#include "SEGGER_RTT.h"


int main (void)
{
    SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);

    uint32_t regValue;
    /* CPUID: 0xE000ED00 */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed00));
    SEGGER_RTT_printf(0, "CPUID: 0x%x\n", regValue);
    SEGGER_RTT_printf(0, "Implementer: 0x%x\n", regValue >> 24);
    SEGGER_RTT_printf(0, "Variant 0x%x\n", (regValue & 0x00ff'ffff) >> 20);
    SEGGER_RTT_printf(0, "Architecture 0x%x\n", (regValue & 0x000f'ffff) >> 16);
    SEGGER_RTT_printf(0, "PartNo 0x%x\n", (regValue & 0x0000'ffff) >> 4);
    SEGGER_RTT_printf(0, "Revision 0x%x\n", (regValue & 0x0000'000f));
 
    /* REVIDR: 0xE000ECFC */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ecfc));
    SEGGER_RTT_printf(0, "CPUID: 0x%x\n", regValue);

    /* ACTLR: 0xE00E008 */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'e008));
    SEGGER_RTT_printf(0, "ACTLR: 0x%x\n", regValue);

    /* PFCR: */

    /* FPDSCR */

    /* FPSCR */

    /* BreakPoint Unit (BPU) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe00f'f008));
    SEGGER_RTT_printf(0, "BPU: 0x%x\n", regValue);

    /* Instrumentation Trace Macrocell (ITM) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe00f'f00c));
    SEGGER_RTT_printf(0, "ITM: 0x%x\n", regValue);
    
    /* Embedded Trace Macrocell (ITM) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe00f'f00c));
    SEGGER_RTT_printf(0, "ETM: 0x%x\n", regValue);
    
    /* Performance Monitoring Unit (PMU) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe00f'f018));
    SEGGER_RTT_printf(0, "PMU: 0x%x\n", regValue);

    /* Memory Model Feature Register 0 (ID_MMFR0) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed50));
    SEGGER_RTT_printf(0, "ID_MMFR0: 0x%x\n", regValue);

    /* Memory Model Feature Register 3 (ID_MMFR3) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed5c));
    SEGGER_RTT_printf(0, "ID_MMFR3: 0x%x\n", regValue);
    
    /* Instruction Set Attribute Register 0 (ID_ISAR0) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed60));
    SEGGER_RTT_printf(0, "ID_ISAR0: 0x%x\n", regValue);

    /* Cache Level ID Register (CLIDR) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed78));
    SEGGER_RTT_printf(0, "CLIDR: 0x%x\n", regValue);

    /* Cache Type Register (CTR) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed7c));
    SEGGER_RTT_printf(0, "CTR: 0x%x\n", regValue);

    /* Cache Size Selection Register (CSSELR) */
    // *(reinterpret_cast<volatile uint32_t *>(0xe00'ed84)) = 0x0000'0000;

    /* Current Cache Size ID Register (CSIDR) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ed80));
    SEGGER_RTT_printf(0, "CSIDR: 0x%x\n", regValue);

    /* Media and VFP Feature Register 0 (MVFR0) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ef40));
    SEGGER_RTT_printf(0, "MVFR0: 0x%x\n", regValue);

    /* Media and VFP Feature Register 1 (MVFR1) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ef44));
    SEGGER_RTT_printf(0, "MVFR1: 0x%x\n", regValue);

    /* Media and VFP Feature Register 2 (MVFR2) */
    regValue = *(reinterpret_cast<volatile uint32_t *>(0xe000'ef48));
    SEGGER_RTT_printf(0, "MVFR2: 0x%x\n", regValue);


    while (1) __WFI();
}
