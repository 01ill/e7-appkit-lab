/* Copyright (C) 2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
*/
// #include <arm_mve_types.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ratio>

#include "board.h"
#include "RTE_Components.h"
#include "m-profile/cmsis_armclang_m.h"
#include CMSIS_device_header

#include "fault_handler.h"

#include "benchmark.hpp"
#include "timing.hpp"
#include <arm_mve.h>
#include "arm_math.h"
#include "SEGGER_RTT.h"

static char PRINTF_OUT_STRING[256] __attribute__((used, section(".bss.array_region_sram0")));


static constexpr uint32_t arrayMaxSize = 48;
static float32_t bigA[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigB[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigC[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t bigCRef[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t packedA[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));
static float32_t packedB[arrayMaxSize*arrayMaxSize] __attribute__((used, section(".bss.array_region_sram0")));

extern "C" {
    float32_t gemm_4x6(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len_k);
    float32_t gemm_4x4(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len_k);
    float32_t gemm_asm_4x4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_4x6(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len);
    float32_t gemm_asm_4x7(const float32_t * a, const float32_t * b, float32_t * c, const uint32_t len);
}

float32_t gemm_reference_4x4(const float32_t *__restrict__ a, const float32_t *__restrict__ b, float32_t * __restrict__ c, const uint32_t len_k) {
    const uint32_t len_n = 4;
    const uint32_t len_m = 4;
    for (uint32_t k = 0; k < len_k; k++) {
        for (uint32_t n = 0; n < len_n; n++) {
            for (uint32_t m = 0; m < len_m; m++) {
                c[m * len_n + n] += a[m * len_k + k] * b[k * len_n + n];
            }
        }    
    }
    return c[0];
}

float32_t gemm_reference_4x6(const float32_t *__restrict__ a, const float32_t *__restrict__ b, float32_t * __restrict__ c, const uint32_t len_k) {
    const uint32_t len_n = 6;
    const uint32_t len_m = 4;
    for (uint32_t k = 0; k < len_k; k++) {
        for (uint32_t n = 0; n < len_n; n++) {
            for (uint32_t m = 0; m < len_m; m++) {
                c[m * len_n + n] += a[m * len_k + k] * b[k * len_n + n];
            }
        }
    }
    return c[0];
}

/*
M=N=K=len A: mxk, B: kxn, C: mxn
Cij = sum Aik * Bkj
*/
float32_t gemm_reference(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) { // i = m
        for (uint32_t j = 0; j < len; j++) { // j = n
            for (uint32_t k = 0; k < len; k++) { // k = k
                c[i * len + j] += a[i * len + k] * b[k * len + j];
            }
        }
    }
    return c[0];
}

float32_t gemm_reorder(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        for (uint32_t k = 0; k < len; k++) {
            for (uint32_t j = 0; j < len; j++) {
                c[i * len + j] += a[i * len + k] * b[k * len + j];
            }
        }
    }
    return c[0];
}

float32_t gemm_blocked_k(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    constexpr uint32_t blockSize = 16;
    for (uint32_t kBlock = 0; kBlock < len; kBlock += blockSize) {
        for (uint32_t i = 0; i < len; i++) {
            uint32_t kBlockEnd = kBlock + blockSize; // kein min nötig weil eh nur 2er Potenzen genutzt werden
            for (uint32_t k = kBlock; k < kBlockEnd; k++) {
                for (uint32_t j = 0; j < len; j++) {
                    c[i * len + j] += a[i * len + k] * b[k * len + j];
                }
            }
        }    
    }

    return c[0];
}

/*
M=N=K=len A: mxk, B: kxn, C: mxn
Cij = sum Aik * Bkj
*/
float32_t gemm_reference_column_major(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j++) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            for (uint32_t k = 0; k < len; k++) { // k = k
                c[j * len + i] += a[k * len + i] * b[j * len + k];
            }
        }
    }
    return c[0];
}

void addDot(uint32_t k, const float32_t * __restrict__ x, const float32_t * __restrict__ y, float32_t * __restrict__ result, uint32_t len) {
    for (uint32_t p = 0; p < k; p++) {
        *result += x[p * len] * y[p];
    }
}

/*
M=N=K=len A: mxk, B: kxn, C: mxn
Cij = sum Aik * Bkj
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization1
*/
float32_t gemm_cm_dot(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j++) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot(len, &a[i], &b[j * len], &c[j * len + i], len); // innerste Schleife mit Dot Product ersetzen
        }
    }
    return c[0];
}

/* https://github.com/flame/how-to-optimize-gemm/wiki/Optimization2 */
float32_t gemm_cm_dot_unroll4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot(len, &a[i], &b[j * len], &c[j * len + i], len); // innerste Schleife mit Dot Product ersetzen
            addDot(len, &a[0], &b[(j+1) * len], &c[(j+1) * len + i], len);
            addDot(len, &a[0], &b[(j+2) * len], &c[(j+2) * len + i], len);
            addDot(len, &a[0], &b[(j+3) * len], &c[(j+3) * len + i], len);
        }
    }
    return c[0];
}

/*
Berechnet C[0,0:3]
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_3
*/
void addDot1x4(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    addDot(k, &a[0], &b[0], &c[0], len);
    addDot(k, &a[0], &b[1 * len], &c[1 * len], len);
    addDot(k, &a[0], &b[2 * len], &c[2 * len], len);
    addDot(k, &a[0], &b[3 * len], &c[3 * len], len);
}

float32_t gemm_cm_dot_unroll1x4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/* https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_4 
// wie addDot1x4 aber ohne weitere Funktionsaufrufe
// */
void addDot1x4_inline(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    for (uint32_t p = 0; p < k; p++) {
        c[0] += a[p * len] * b[p];
    }
    for (uint32_t p = 0; p < k; p++) {
        c[1*len] += a[p * len] * b[p + 1*len];
    }
    for (uint32_t p = 0; p < k; p++) {
        c[2*len] += a[p * len] * b[p + 2*len];
    }
    for (uint32_t p = 0; p < k; p++) {
        c[3*len] += a[p * len] * b[p + 3*len];
    }
}

float32_t gemm_cm_dot_unroll1x4_inline(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_5
*/
void addDot1x4_inline_fused(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    for (uint32_t p = 0; p < k; p++) {
        c[0] += a[p * len] * b[p];
        c[1*len] += a[p * len] * b[p + 1*len];
        c[2*len] += a[p * len] * b[p + 2*len];
        c[3*len] += a[p * len] * b[p + 3*len];
    }
}

float32_t gemm_cm_dot_unroll1x4_inline_fused(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline_fused(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_6
*/
void addDot1x4_inline_fused_accumulate(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // register nicht mehr erlaubt ...
    float32_t c00_reg = 0.0, c01_reg = 0.0, c02_reg = 0.0, c03_reg = 0.0, a0p_reg = 0.0;

    for (uint32_t p = 0; p < k; p++) {
        a0p_reg = a[p * len];

        c00_reg += a0p_reg * b[p];
        c01_reg += a0p_reg * b[p + 1*len];
        c02_reg += a0p_reg * b[p + 2*len];
        c03_reg += a0p_reg * b[p + 3*len];
    }
    c[0] += c00_reg;
    c[len] += c01_reg;
    c[2*len] = c02_reg;
    c[3*len] = c03_reg;
}

float32_t gemm_cm_dot_unroll1x4_inline_fused_accumulate(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline_fused_accumulate(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_7
*/
void addDot1x4_inline_fused_accumulate_pointers(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // register nicht mehr erlaubt ...
    float32_t c00_reg = 0.0, c01_reg = 0.0, c02_reg = 0.0, c03_reg = 0.0, a0p_reg = 0.0;
    const float32_t *bp0_p, *bp1_p, *bp2_p, *bp3_p;
    bp0_p = &b[0];
    bp1_p = &b[len];
    bp2_p = &b[2*len];
    bp3_p = &b[3*len];

    for (uint32_t p = 0; p < k; p++) {
        a0p_reg = a[p * len];

        c00_reg += a0p_reg * *bp0_p++;
        c01_reg += a0p_reg * *bp1_p++;
        c02_reg += a0p_reg * *bp2_p++;
        c03_reg += a0p_reg * *bp3_p++;
    }
    c[0] += c00_reg;
    c[len] += c01_reg;
    c[2*len] = c02_reg;
    c[3*len] = c03_reg;
}

float32_t gemm_cm_dot_unroll1x4_inline_fused_accumulate_pointers(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline_fused_accumulate_pointers(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

void addDot1x4_inline_fused_accumulate_pointers_unroll4(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // register nicht mehr erlaubt ...
    float32_t c00_reg = 0.0, c01_reg = 0.0, c02_reg = 0.0, c03_reg = 0.0, a0p_reg = 0.0;
    const float32_t *bp0_p, *bp1_p, *bp2_p, *bp3_p;
    bp0_p = &b[0];
    bp1_p = &b[len];
    bp2_p = &b[2*len];
    bp3_p = &b[3*len];

    for (uint32_t p = 0; p < k; p += 4) {
        a0p_reg = a[p * len];

        c00_reg += a0p_reg * *bp0_p++;
        c01_reg += a0p_reg * *bp1_p++;
        c02_reg += a0p_reg * *bp2_p++;
        c03_reg += a0p_reg * *bp3_p++;

        a0p_reg = a[(p+1) * len];

        c00_reg += a0p_reg * *bp0_p++;
        c01_reg += a0p_reg * *bp1_p++;
        c02_reg += a0p_reg * *bp2_p++;
        c03_reg += a0p_reg * *bp3_p++;
        
        a0p_reg = a[(p+2) * len];

        c00_reg += a0p_reg * *bp0_p++;
        c01_reg += a0p_reg * *bp1_p++;
        c02_reg += a0p_reg * *bp2_p++;
        c03_reg += a0p_reg * *bp3_p++;
        
        a0p_reg = a[(p+3) * len];

        c00_reg += a0p_reg * *bp0_p++;
        c01_reg += a0p_reg * *bp1_p++;
        c02_reg += a0p_reg * *bp2_p++;
        c03_reg += a0p_reg * *bp3_p++;
    }
    c[0] += c00_reg;
    c[len] += c01_reg;
    c[2*len] = c02_reg;
    c[3*len] = c03_reg;
}

float32_t gemm_cm_dot_unroll1x4_inline_fused_accumulate_pointers_unroll4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline_fused_accumulate_pointers_unroll4(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_9
*/
void addDot1x4_inline_fused_accumulate_pointers_unroll4_no_writeback(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // register nicht mehr erlaubt ...
    float32_t c00_reg = 0.0, c01_reg = 0.0, c02_reg = 0.0, c03_reg = 0.0, a0p_reg = 0.0;
    const float32_t *bp0_p, *bp1_p, *bp2_p, *bp3_p;
    bp0_p = &b[0];
    bp1_p = &b[len];
    bp2_p = &b[2*len];
    bp3_p = &b[3*len];

    for (uint32_t p = 0; p < k; p += 4) {
        a0p_reg = a[p * len];

        c00_reg += a0p_reg * *bp0_p;
        c01_reg += a0p_reg * *bp1_p;
        c02_reg += a0p_reg * *bp2_p;
        c03_reg += a0p_reg * *bp3_p;

        a0p_reg = a[(p+1) * len];

        c00_reg += a0p_reg * *(bp0_p+1);
        c01_reg += a0p_reg * *(bp1_p+1);
        c02_reg += a0p_reg * *(bp2_p+1);
        c03_reg += a0p_reg * *(bp3_p+1);
        
        a0p_reg = a[(p+2) * len];

        c00_reg += a0p_reg * *(bp0_p+2);
        c01_reg += a0p_reg * *(bp1_p+2);
        c02_reg += a0p_reg * *(bp2_p+2);
        c03_reg += a0p_reg * *(bp3_p+2);
        
        a0p_reg = a[(p+3) * len];

        c00_reg += a0p_reg * *(bp0_p+3);
        c01_reg += a0p_reg * *(bp1_p+3);
        c02_reg += a0p_reg * *(bp2_p+3);
        c03_reg += a0p_reg * *(bp3_p+3);

        bp0_p += 4;
        bp1_p += 4;
        bp2_p += 4;
        bp3_p += 4;
    }
    c[0] += c00_reg;
    c[len] += c01_reg;
    c[2*len] = c02_reg;
    c[3*len] = c03_reg;
}


float32_t gemm_cm_dot_unroll1x4_inline_fused_accumulate_pointers_unroll4_no_writeback(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i++) { // i = m
            addDot1x4_inline_fused_accumulate_pointers_unroll4_no_writeback(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_3
*/
void addDot4x4(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {

    // erste Zeile
    addDot(k, &a[0], &b[0], &c[0], len);
    addDot(k, &a[0], &b[1 * len], &c[1 * len], len);
    addDot(k, &a[0], &b[2 * len], &c[2 * len], len);
    addDot(k, &a[0], &b[3 * len], &c[3 * len], len);

    // zweite Zeile
    addDot(k, &a[1], &b[0], &c[1], len);
    addDot(k, &a[1], &b[1 * len], &c[1 * len + 1], len);
    addDot(k, &a[1], &b[2 * len], &c[2 * len + 1], len);
    addDot(k, &a[1], &b[3 * len], &c[3 * len + 1], len);

    // dritte Zeile
    addDot(k, &a[2], &b[0], &c[1], len);
    addDot(k, &a[2], &b[1 * len], &c[1 * len + 2], len);
    addDot(k, &a[2], &b[2 * len], &c[2 * len + 2], len);
    addDot(k, &a[2], &b[3 * len], &c[3 * len + 2], len);
    
    // vierte Zeile
    addDot(k, &a[3], &b[0], &c[1], len);
    addDot(k, &a[3], &b[1 * len], &c[1 * len + 3], len);
    addDot(k, &a[3], &b[2 * len], &c[2 * len + 3], len);
    addDot(k, &a[3], &b[3 * len], &c[3 * len + 3], len);
}


float32_t gemm_cm_dot_unroll4x4(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_4
*/
void addDot4x4_unroll(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {

    // erste Zeile
    // addDot(k, &a[0], &b[0], &c[0], len);
    for (uint32_t p = 0; p < k; p++) {
        c[0] += a[p * len] * b[p];
    }
    // addDot(k, &a[0], &b[1 * len], &c[1 * len], len);
    for (uint32_t p = 0; p < k; p++) {
        c[1*len] += a[p * len] * b[len + p];
    }
    // addDot(k, &a[0], &b[2 * len], &c[2 * len], len);
    for (uint32_t p = 0; p < k; p++) {
        c[2*len] += a[p * len] * b[2*len + p];
    }
    // addDot(k, &a[0], &b[3 * len], &c[3 * len], len);
    for (uint32_t p = 0; p < k; p++) {
        c[3*len] += a[p * len] * b[3*len + p];
    }
    
    // zweite Zeile
    // addDot(k, &a[1], &b[0], &c[1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[1] += a[p * len + 1] * b[p];
    }
    // addDot(k, &a[1], &b[1 * len], &c[1 * len + 1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[len + 1] += a[p * len + 1] * b[len + p];
    }
    // addDot(k, &a[1], &b[2 * len], &c[2 * len + 1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[2*len + 1] += a[p * len + 1] * b[2*len + p];
    }
    // addDot(k, &a[1], &b[3 * len], &c[3 * len + 1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[3*len + 1] += a[p * len + 1] * b[3*len + p];
    }

    // dritte Zeile
    // addDot(k, &a[2], &b[0], &c[1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[2] += a[p * len + 2] * b[p];
    }
    // addDot(k, &a[2], &b[1 * len], &c[1 * len + 2], len);
    for (uint32_t p = 0; p < k; p++) {
        c[len + 2] += a[p * len + 2] * b[len + p];
    }
    // addDot(k, &a[2], &b[2 * len], &c[2 * len + 2], len);
    for (uint32_t p = 0; p < k; p++) {
        c[2*len + 2] += a[p * len + 2] * b[2*len + p];
    }
    // addDot(k, &a[2], &b[3 * len], &c[3 * len + 2], len);
    for (uint32_t p = 0; p < k; p++) {
        c[3*len + 2] += a[p * len + 2] * b[3*len + p];
    }

    // vierte Zeile
    // addDot(k, &a[3], &b[0], &c[1], len);
    for (uint32_t p = 0; p < k; p++) {
        c[3] += a[p * len + 3] * b[p];
    }
    // addDot(k, &a[3], &b[1 * len], &c[1 * len + 3], len);
    for (uint32_t p = 0; p < k; p++) {
        c[len + 3] += a[p * len + 3] * b[len + p];
    }
    // addDot(k, &a[3], &b[2 * len], &c[2 * len + 3], len);
    for (uint32_t p = 0; p < k; p++) {
        c[2*len + 3] += a[p * len + 3] * b[2*len + p];
    }
    // addDot(k, &a[3], &b[3 * len], &c[3 * len + 3], len);
    for (uint32_t p = 0; p < k; p++) {
        c[3*len + 3] += a[p * len + 3] * b[3*len + p];
    }
}


float32_t gemm_cm_dot_unroll4x4_unroll(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_5
*/
void addDot4x4_unroll_fused(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {

    // erste Zeile
    // addDot(k, &a[0], &b[0], &c[0], len);
    for (uint32_t p = 0; p < k; p++) {
        c[0] += a[p * len] * b[p];
    // addDot(k, &a[0], &b[1 * len], &c[1 * len], len);
        c[1*len] += a[p * len] * b[len + p];
    // addDot(k, &a[0], &b[2 * len], &c[2 * len], len);
        c[2*len] += a[p * len] * b[2*len + p];
    // addDot(k, &a[0], &b[3 * len], &c[3 * len], len);
        c[3*len] += a[p * len] * b[3*len + p];
    
    // zweite Zeile
    // addDot(k, &a[1], &b[0], &c[1], len);
        c[1] += a[p * len + 1] * b[p];
    // addDot(k, &a[1], &b[1 * len], &c[1 * len + 1], len);
        c[len + 1] += a[p * len + 1] * b[len + p];
    // addDot(k, &a[1], &b[2 * len], &c[2 * len + 1], len);
        c[2*len + 1] += a[p * len + 1] * b[2*len + p];
    // addDot(k, &a[1], &b[3 * len], &c[3 * len + 1], len);
        c[3*len + 1] += a[p * len + 1] * b[3*len + p];

    // dritte Zeile
    // addDot(k, &a[2], &b[0], &c[1], len);
        c[2] += a[p * len + 2] * b[p];
    // addDot(k, &a[2], &b[1 * len], &c[1 * len + 2], len);
        c[len + 2] += a[p * len + 2] * b[len + p];
    // addDot(k, &a[2], &b[2 * len], &c[2 * len + 2], len);
        c[2*len + 2] += a[p * len + 2] * b[2*len + p];
    // addDot(k, &a[2], &b[3 * len], &c[3 * len + 2], len);
        c[3*len + 2] += a[p * len + 2] * b[3*len + p];

    // vierte Zeile
    // addDot(k, &a[3], &b[0], &c[1], len);
        c[3] += a[p * len + 3] * b[p];
    // addDot(k, &a[3], &b[1 * len], &c[1 * len + 3], len);
        c[len + 3] += a[p * len + 3] * b[len + p];
    // addDot(k, &a[3], &b[2 * len], &c[2 * len + 3], len);
        c[2*len + 3] += a[p * len + 3] * b[2*len + p];
    // addDot(k, &a[3], &b[3 * len], &c[3 * len + 3], len);
        c[3*len + 3] += a[p * len + 3] * b[3*len + p];
    }
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_6
*/
void addDot4x4_unroll_fused_accumulate(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    float32_t c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0,
        c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0,
        c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0,
        c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0,
        a0p = 0.0, a1p = 0.0, a2p = 0.0, a3p = 0.0;

    for (uint32_t p = 0; p < k; p++) {
        a0p = a[len * p];
        a1p = a[p*len + 1];
        a2p = a[p*len + 2];
        a3p = a[p*len + 3];

        c00 += a0p * b[p];
        c01 += a0p * b[len + p];
        c02 += a0p * b[2*len + p];
        c03 += a0p * b[3*len + p];

        c10 += a1p * b[p];
        c11 += a1p * b[len + p];
        c12 += a1p * b[2*len + p];
        c13 += a1p * b[3*len + p];
        
        c20 += a2p * b[p];
        c21 += a2p * b[len + p];
        c02 += a2p * b[2*len + p];
        c23 += a2p * b[3*len + p];
        
        c30 += a3p * b[p];
        c31 += a3p * b[len + p];
        c32 += a3p * b[2*len + p];
        c33 += a3p * b[3*len + p];
    }

    c[0] += c00;
    c[len] += c01;
    c[2*len] += c02;
    c[3*len] += c03;

    c[1] += c10;
    c[1 + len] += c11;
    c[1 + 2*len] += c12;
    c[1 + 3*len] += c13;

    c[2] += c20;
    c[2 + len] += c21;
    c[2 + 2*len] += c22;
    c[2 + 3*len] += c23;

    c[3] += c30;
    c[3 + len] += c31;
    c[3 + 2*len] += c32;
    c[3 + 3*len] += c33;
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_7
*/
void addDot4x4_unroll_fused_accumulate_pointers(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    float32_t c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0,
        c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0,
        c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0,
        c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0,
        a0p = 0.0, a1p = 0.0, a2p = 0.0, a3p = 0.0;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    for (uint32_t p = 0; p < k; p++) {
        a0p = a[len * p];
        a1p = a[p*len + 1];
        a2p = a[p*len + 2];
        a3p = a[p*len + 3];

        bp0 = &b[p];
        bp1 = &b[len + p];
        bp2 = &b[2*len + p];
        bp3 = &b[3*len + p];

        c00 += a0p * *bp0;
        c01 += a0p * *bp1;
        c02 += a0p * *bp2;
        c03 += a0p * *bp3;

        c10 += a1p * *bp0;
        c11 += a1p * *bp1;
        c12 += a1p * *bp2;
        c13 += a1p * *bp3;
        
        c20 += a2p * *bp0;
        c21 += a2p * *bp1;
        c02 += a2p * *bp2;
        c23 += a2p * *bp3;
        
        c30 += a3p * *bp0++;
        c31 += a3p * *bp1++;
        c32 += a3p * *bp2++;
        c33 += a3p * *bp3++;
    }

    c[0] += c00;
    c[len] += c01;
    c[2*len] += c02;
    c[3*len] += c03;

    c[1] += c10;
    c[1 + len] += c11;
    c[1 + 2*len] += c12;
    c[1 + 3*len] += c13;

    c[2] += c20;
    c[2 + len] += c21;
    c[2 + 2*len] += c22;
    c[2 + 3*len] += c23;

    c[3] += c30;
    c[3 + len] += c31;
    c[3 + 2*len] += c32;
    c[3 + 3*len] += c33;
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
*/
void addDot4x4_unroll_fused_accumulate_pointers_v2(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    float32_t c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0,
        c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0,
        c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0,
        c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0,
        a0p = 0.0, a1p = 0.0, a2p = 0.0, a3p = 0.0,
        bp0_r, bp1_r, bp2_r, bp3_r;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];


    for (uint32_t p = 0; p < k; p++) {
        a0p = a[len * p];
        a1p = a[p*len + 1];
        a2p = a[p*len + 2];
        a3p = a[p*len + 3];

        bp0_r = *bp0++;
        bp1_r = *bp1++;
        bp2_r = *bp2++;
        bp3_r = *bp3++;

        c00 += a0p * bp0_r;
        c01 += a0p * bp1_r;
        c02 += a0p * bp2_r;
        c03 += a0p * bp3_r;

        c10 += a1p * bp0_r;
        c11 += a1p * bp1_r;
        c12 += a1p * bp2_r;
        c13 += a1p * bp3_r;
        
        c20 += a2p * bp0_r;
        c21 += a2p * bp1_r;
        c02 += a2p * bp2_r;
        c23 += a2p * bp3_r;
        
        c30 += a3p * bp0_r;
        c31 += a3p * bp1_r;
        c32 += a3p * bp2_r;
        c33 += a3p * bp3_r;
    }

    c[0] += c00;
    c[len] += c01;
    c[2*len] += c02;
    c[3*len] += c03;

    c[1] += c10;
    c[1 + len] += c11;
    c[1 + 2*len] += c12;
    c[1 + 3*len] += c13;

    c[2] += c20;
    c[2 + len] += c21;
    c[2 + 2*len] += c22;
    c[2 + 3*len] += c23;

    c[3] += c30;
    c[3 + len] += c31;
    c[3 + 2*len] += c32;
    c[3 + 3*len] += c33;
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_9
*/
void addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    float32_t c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0,
        c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0,
        c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0,
        c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0,
        a0p = 0.0, a1p = 0.0, a2p = 0.0, a3p = 0.0,
        bp0_r, bp1_r, bp2_r, bp3_r;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];


    for (uint32_t p = 0; p < k; p++) {
        a0p = a[len * p];
        a1p = a[p*len + 1];
        a2p = a[p*len + 2];
        a3p = a[p*len + 3];

        bp0_r = *bp0++;
        bp1_r = *bp1++;
        bp2_r = *bp2++;
        bp3_r = *bp3++;

        // Es werden jetzt immer zwei Reihen zugleich berechnet. Dann geht es später einfach mit den Vektor-Registern
        c00 += a0p * bp0_r;
        c10 += a1p * bp0_r;

        c01 += a0p * bp1_r;
        c11 += a1p * bp1_r;

        c02 += a0p * bp2_r;
        c12 += a1p * bp2_r;

        c03 += a0p * bp3_r;
        c13 += a1p * bp3_r;
        
        c20 += a2p * bp0_r;
        c30 += a3p * bp0_r;

        c21 += a2p * bp1_r;
        c31 += a3p * bp1_r;

        c02 += a2p * bp2_r;
        c32 += a3p * bp2_r;

        c23 += a2p * bp3_r;
        c33 += a3p * bp3_r;
    }

    c[0] += c00;
    c[len] += c01;
    c[2*len] += c02;
    c[3*len] += c03;

    c[1] += c10;
    c[1 + len] += c11;
    c[1 + 2*len] += c12;
    c[1 + 3*len] += c13;

    c[2] += c20;
    c[2 + len] += c21;
    c[2 + 2*len] += c22;
    c[2 + 3*len] += c23;

    c[3] += c30;
    c[3 + len] += c31;
    c[3 + 2*len] += c32;
    c[3 + 3*len] += c33;
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_10
*/
void addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const uint32_t lda, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, a0p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];

    // Vektorregister initialisieren mit vdup
    c0_vreg = vdupq_n_f32(0.0f);
    c1_vreg = vdupq_n_f32(0.0f);
    c2_vreg = vdupq_n_f32(0.0f);
    c3_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);

        c0_vreg = vfmaq_f32(c0_vreg, a0p_vreg, bp0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, a0p_vreg, bp1_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, a0p_vreg, bp2_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, a0p_vreg, bp3_vreg);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[len], c1_vreg);
    vst1q_f32(&c[len*2], c2_vreg);
    vst1q_f32(&c[len*3], c3_vreg);
}

void addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics2(uint32_t k, const float32_t * __restrict__ a, const uint32_t lda, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg, a0p_vreg;
       // bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];

    // Vektorregister initialisieren mit vdup
    c0_vreg = vdupq_n_f32(0.0f);
    c1_vreg = vdupq_n_f32(0.0f);
    c2_vreg = vdupq_n_f32(0.0f);
    c3_vreg = vdupq_n_f32(0.0f);

    #pragma unroll
    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        /*bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);*/

        c0_vreg = vfmaq_n_f32(c0_vreg, a0p_vreg, *bp0++);
        c1_vreg = vfmaq_n_f32(c1_vreg, a0p_vreg, *bp1++);
        c2_vreg = vfmaq_n_f32(c2_vreg, a0p_vreg, *bp2++);
        c3_vreg = vfmaq_n_f32(c3_vreg, a0p_vreg, *bp3++);
    }

    vst1q_f32(&c[0], c0_vreg);
    vst1q_f32(&c[len], c1_vreg);
    vst1q_f32(&c[len*2], c2_vreg);
    vst1q_f32(&c[len*3], c3_vreg);
}


float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], len, &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics2(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics2(len, &a[i], len, &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}


/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_11
*/
void inner_kernel_4x4_intrinsics(uint32_t m, uint32_t n, uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, uint32_t len) {
    for (uint32_t j = 0; j < n; j += 4) {
        for (uint32_t i = 0; i < m; i += 4) {
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &a[i], len, &b[j * len], &c[j * len + i], len);
        }
    }
}

constexpr uint32_t mc = 16;
constexpr uint32_t kc = 16;

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics(ib, len, pb, &a[p*len + i], &b[p], &c[i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_12
*/
void packA(uint32_t k, const float32_t * __restrict__ a, uint32_t len, float32_t * __restrict__ aPack) {
    for (uint32_t j = 0; j < k; j++) {
        const float32_t *a_ij_p = &a[j * len];

        *aPack++ = *a_ij_p;
        *aPack++ = *(a_ij_p+1);
        *aPack++ = *(a_ij_p+2);
        *aPack++ = *(a_ij_p+3);
    }
}

void inner_kernel_4x4_intrinsics_packed(const uint32_t m, const uint32_t n, const uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < n; j += 4) {
        for (uint32_t i = 0; i < m; i += 4) {
            packA(k, &a[i], len, &packedA[i*k]);
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &packedA[i*k], 4, &b[j * len], &c[j * len + i], len);
        }
    }
}

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked_packed(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics_packed(ib, len, pb, &a[p*len + i], &b[p], &c[i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_13
*/
void inner_kernel_4x4_intrinsics_packed_optimized(const uint32_t m, const uint32_t n, const uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < n; j += 4) {
        for (uint32_t i = 0; i < m; i += 4) {
            if (j == 0) packA(k, &a[i], len, &packedA[i*k]);
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &packedA[i*k], 4, &b[j * len], &c[j * len + i], len);
        }
    }
}

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked_packed_optimized(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics_packed_optimized(ib, len, pb, &a[p*len + i], &b[p], &c[i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_14
*/
void packB(uint32_t k, const float32_t * __restrict__ b, uint32_t len, float32_t *bPack) {
    const float32_t *bi0 = &b[0], *bi1 = &b[len], *bi2 = &b[2*len], *bi3 = &b[3*len];
    for (uint32_t j = 0; j < k; j++) {
        *bPack++ = *bi0++;
        *bPack++ = *bi1++;
        *bPack++ = *bi2++;
        *bPack++ = *bi3++;
    }
}

void inner_kernel_4x4_intrinsics_packed_ab(const uint32_t m, const uint32_t n, const uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < n; j += 4) {
        packB(k, &b[j*len], len, &packedB[j*k]);
        for (uint32_t i = 0; i < m; i += 4) {
            if (j == 0) packA(k, &a[i], len, &packedA[i*k]);
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &packedA[i*k], 4, &packedB[j*k], &c[j * len + i], len);
        }
    }
}

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked_packed_ab(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics_packed_ab(ib, len, pb, &a[p*len + i], &b[p], &c[i], len);
        }
    }
    return c[0];
}

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_15
*/
void inner_kernel_4x4_intrinsics_packed_ab_optimized(const uint32_t m, const uint32_t n, const uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b,  float32_t * __restrict__ c, const uint32_t len, bool firstTime) {
    for (uint32_t j = 0; j < n; j += 4) {
        if (firstTime) packB(k, &b[j*len], len, &packedB[j*k]);
        for (uint32_t i = 0; i < m; i += 4) {
            if (j == 0) packA(k, &a[i], len, &packedA[i*k]);
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(k, &packedA[i*k], 4, &packedB[j*k], &c[j * len + i], len);
        }
    }
}

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics_blocked_packed_ab_optimized(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t p = 0; p < len; p += kc) {
        uint32_t pb = std::min(len-p, kc);
        for (uint32_t i = 0; i < len; i += mc) {
            uint32_t ib = std::min(len - i, mc);
            inner_kernel_4x4_intrinsics_packed_ab_optimized(ib, len, pb, &a[p*len + i], &b[p], &c[i], len, i==0);
        }
    }
    return c[0];
}

__NO_RETURN int main (void) {
    fault_dump_enable(true);
    SEGGER_RTT_ConfigUpBuffer(0, nullptr, nullptr, 0, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
    setupTests();

    uint32_t iterations = 1000;
    uint32_t time;
    float32_t gflops;
    float32_t result = 0.0f;

    arm_matrix_instance_f32 armA;
    arm_matrix_instance_f32 armB;
    arm_matrix_instance_f32 armC;
    arm_status status;

    for (uint32_t i = 48; i <= arrayMaxSize; i += 48) {
        for (uint32_t j = 0; j < i*i; j++) {
            bigA[j] = j;
            bigB[j] = j;
            bigC[j] = 0;
        }
        arm_mat_init_f32(&armA, i, i, bigA);
        arm_mat_init_f32(&armB, i, i, bigB);
        arm_mat_init_f32(&armC, i, i, bigC);

        time = benchmarkArm(arm_mat_mult_f32, iterations, &status, &armA, &armB, &armC);
        gflops = ((iterations * 2 * pow(i,3) * 1000.0f)  / time) / pow(10, 9);
        sprintf(PRINTF_OUT_STRING, "CMSIS-DSP %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


        for (uint32_t j = 0; j < i*i; j++) {
            bigA[j] = j;
            bigB[j] = j;
            bigC[j] = 0;
        }
        time = benchmark(gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics, iterations, &result, bigA, bigB, bigC, i);
        gflops = ((iterations * 2 * pow(i,3) * 1000.0f)  / time) / pow(10, 9);
        sprintf(PRINTF_OUT_STRING, "GEMM CM Intrinsics %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

        for (uint32_t j = 0; j < i*i; j++) {
            bigA[j] = j;
            bigB[j] = j;
            bigC[j] = 0;
        }
        RTC_Clock::time_point start = RTC_Clock::now();
        for (uint32_t j = 0; j < iterations; j++) {
            gemm_asm_4x4(bigA, bigB, bigC, i);
        }
        RTC_Clock::time_point end = RTC_Clock::now();
    
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        gflops = ((iterations * 2 * pow(i,3) * 1000.0f)  / time) / pow(10, 9);
        sprintf(PRINTF_OUT_STRING, "GEMM CM 4x4 ASM %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);

        for (uint32_t j = 0; j < i*i; j++) {
            bigA[j] = j;
            bigB[j] = j;
            bigC[j] = 0;
        }
        start = RTC_Clock::now();
        for (uint32_t j = 0; j < iterations; j++) {
            gemm_asm_4x6(bigA, bigB, bigC, i);
        }
        end = RTC_Clock::now();
    
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        gflops = ((iterations * 2 * pow(i,3) * 1000.0f)  / time) / pow(10, 9);
        sprintf(PRINTF_OUT_STRING, "GEMM CM 4x6 ASM %dx%d (%d): %f, %f, %f\r\n", i, i, time, bigC[0], result, gflops);
        SEGGER_RTT_WriteString(0, PRINTF_OUT_STRING);


        //iterations = iterations >> 2U; // Iterationen müssen weniger werden, sonst rechnet der nie fertig
        iterations >>= 1U;
    }
    SEGGER_RTT_printf(0, "Fertig!\n");

    for (uint32_t j = 0; j < 12*12; j++) {
        bigA[j] = j;
        bigB[j] = j;
        bigC[j] = 0;
        bigCRef[j] = 0;
    }
    gemm_asm_4x4(bigA, bigB, bigC, 12);
    gemm_cm_dot(bigA, bigB, bigCRef, 12);
    int32_t compareResult = compare(bigC, bigCRef, 12*12);
    if (compareResult == -1) {
        SEGGER_RTT_printf(0, "Test erfolgreich!\n");
    } else {
        SEGGER_RTT_printf(0, "Test nicht erfolgreich bei %d\n", compareResult);
    }

    stopTests();

    while (1) __WFE();
}

/*
#define TRAP_RET_ZERO  {__BKPT(0); return 0;}
int _close(int val) TRAP_RET_ZERO
int _lseek(int val0, int val1, int val2) TRAP_RET_ZERO
int _read(int val0, char * val1, int val2) TRAP_RET_ZERO
int _write(int val0, char * val1, int val2) TRAP_RET_ZERO
*/