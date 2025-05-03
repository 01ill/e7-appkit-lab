#include <arm_mve.h>

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

float32_t gemm_cm_dot_unroll4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics2(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 4) { // i = m
            addDot4x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics2(len, &a[i], len, &b[j * len], &c[j * len + i], len);
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

/*
https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_10
*/
void addDot8x8_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c00_vreg, c01_vreg, c10_vreg, c11_vreg, c20_vreg, c21_vreg, c30_vreg, c31_vreg, c40_vreg, c41_vreg, c50_vreg, c51_vreg, c60_vreg, c61_vreg, c70_vreg, c71_vreg, a0p_vreg, a1p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg, bp4_vreg, bp5_vreg, bp6_vreg, bp7_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3, *bp4, *bp5, *bp6, *bp7;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];
    bp4 = &b[4*len];
    bp5 = &b[5*len];
    bp6 = &b[6*len];
    bp7 = &b[7*len];

    // Vektorregister initialisieren mit vdup
    c00_vreg = vdupq_n_f32(0.0f);
    c01_vreg = vdupq_n_f32(0.0f);
    c10_vreg = vdupq_n_f32(0.0f);
    c11_vreg = vdupq_n_f32(0.0f);
    c20_vreg = vdupq_n_f32(0.0f);
    c21_vreg = vdupq_n_f32(0.0f);
    c30_vreg = vdupq_n_f32(0.0f);
    c31_vreg = vdupq_n_f32(0.0f);
    c40_vreg = vdupq_n_f32(0.0f);
    c41_vreg = vdupq_n_f32(0.0f);
    c50_vreg = vdupq_n_f32(0.0f);
    c51_vreg = vdupq_n_f32(0.0f);
    c60_vreg = vdupq_n_f32(0.0f);
    c61_vreg = vdupq_n_f32(0.0f);
    c70_vreg = vdupq_n_f32(0.0f);
    c71_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        a1p_vreg = vld1q_f32(&a[p*len+4]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);
        bp4_vreg = vdupq_n_f32(*bp4++);
        bp5_vreg = vdupq_n_f32(*bp5++);
        bp6_vreg = vdupq_n_f32(*bp6++);
        bp7_vreg = vdupq_n_f32(*bp7++);

        c00_vreg = vfmaq_f32(c00_vreg, a0p_vreg, bp0_vreg);
        c01_vreg = vfmaq_f32(c01_vreg, a1p_vreg, bp0_vreg);
        c10_vreg = vfmaq_f32(c10_vreg, a0p_vreg, bp1_vreg);
        c11_vreg = vfmaq_f32(c11_vreg, a1p_vreg, bp1_vreg);
        c20_vreg = vfmaq_f32(c20_vreg, a0p_vreg, bp2_vreg);
        c21_vreg = vfmaq_f32(c21_vreg, a1p_vreg, bp2_vreg);
        c30_vreg = vfmaq_f32(c30_vreg, a0p_vreg, bp3_vreg);
        c31_vreg = vfmaq_f32(c31_vreg, a1p_vreg, bp3_vreg);
        c40_vreg = vfmaq_f32(c40_vreg, a0p_vreg, bp4_vreg);
        c41_vreg = vfmaq_f32(c41_vreg, a1p_vreg, bp4_vreg);
        c50_vreg = vfmaq_f32(c50_vreg, a0p_vreg, bp5_vreg);
        c51_vreg = vfmaq_f32(c51_vreg, a1p_vreg, bp5_vreg);
        c60_vreg = vfmaq_f32(c60_vreg, a0p_vreg, bp6_vreg);
        c61_vreg = vfmaq_f32(c61_vreg, a1p_vreg, bp6_vreg);
        c70_vreg = vfmaq_f32(c70_vreg, a0p_vreg, bp7_vreg);
        c71_vreg = vfmaq_f32(c71_vreg, a1p_vreg, bp7_vreg);

    }

    vst1q_f32(&c[0], c00_vreg);
    vst1q_f32(&c[4], c01_vreg);
    vst1q_f32(&c[len], c10_vreg);
    vst1q_f32(&c[len+4], c11_vreg);
    vst1q_f32(&c[2*len], c20_vreg);
    vst1q_f32(&c[2*len+4], c21_vreg);
    vst1q_f32(&c[3*len], c30_vreg);
    vst1q_f32(&c[3*len+4], c31_vreg);
    vst1q_f32(&c[4*len], c40_vreg);
    vst1q_f32(&c[4*len+4], c41_vreg);
    vst1q_f32(&c[5*len], c50_vreg);
    vst1q_f32(&c[5*len+4], c51_vreg);
    vst1q_f32(&c[6*len], c60_vreg);
    vst1q_f32(&c[6*len+4], c61_vreg);
    vst1q_f32(&c[7*len], c70_vreg);
    vst1q_f32(&c[7*len+4], c71_vreg);
}

float32_t gemm_cm_dot_unroll8x8_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 8) { // j = n
        for (uint32_t i = 0; i < len; i += 8) { // i = m
            addDot8x8_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}

void addDot8x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(uint32_t k, const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, uint32_t len) {
    // Wir wollen Single Precision, damit die Register auch genutzt werden können
    // So passen 4 Float32 Werte rein, statt nur 2 Double
    // Also werden 4 Reihen (alle) gefused
    float32x4_t c00_vreg, c01_vreg, c10_vreg, c11_vreg, c20_vreg, c21_vreg, c30_vreg, c31_vreg, a0p_vreg, a1p_vreg,
        bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg;
    const float32_t *bp0, *bp1, *bp2, *bp3;

    bp0 = &b[0];
    bp1 = &b[len];
    bp2 = &b[2*len];
    bp3 = &b[3*len];

    // Vektorregister initialisieren mit vdup
    c00_vreg = vdupq_n_f32(0.0f);
    c01_vreg = vdupq_n_f32(0.0f);
    c10_vreg = vdupq_n_f32(0.0f);
    c11_vreg = vdupq_n_f32(0.0f);
    c20_vreg = vdupq_n_f32(0.0f);
    c21_vreg = vdupq_n_f32(0.0f);
    c30_vreg = vdupq_n_f32(0.0f);
    c31_vreg = vdupq_n_f32(0.0f);


    for (uint32_t p = 0; p < k; p++) {
        a0p_vreg = vld1q_f32(&a[p*len]);
        bp0_vreg = vdupq_n_f32(*bp0++);
        bp1_vreg = vdupq_n_f32(*bp1++);
        bp2_vreg = vdupq_n_f32(*bp2++);
        bp3_vreg = vdupq_n_f32(*bp3++);

        c00_vreg = vfmaq_f32(c00_vreg, a0p_vreg, bp0_vreg);
        c10_vreg = vfmaq_f32(c10_vreg, a0p_vreg, bp1_vreg);
        c20_vreg = vfmaq_f32(c20_vreg, a0p_vreg, bp2_vreg);
        c30_vreg = vfmaq_f32(c30_vreg, a0p_vreg, bp3_vreg);
        a1p_vreg = vld1q_f32(&a[p*len+4]);
        c01_vreg = vfmaq_f32(c01_vreg, a1p_vreg, bp0_vreg);
        c11_vreg = vfmaq_f32(c11_vreg, a1p_vreg, bp1_vreg);
        c21_vreg = vfmaq_f32(c21_vreg, a1p_vreg, bp2_vreg);
        c31_vreg = vfmaq_f32(c31_vreg, a1p_vreg, bp3_vreg);
    }

    vst1q_f32(&c[0], c00_vreg);
    vst1q_f32(&c[4], c01_vreg);
    vst1q_f32(&c[len], c10_vreg);
    vst1q_f32(&c[len+4], c11_vreg);
    vst1q_f32(&c[2*len], c20_vreg);
    vst1q_f32(&c[2*len+4], c21_vreg);
    vst1q_f32(&c[3*len], c30_vreg);
    vst1q_f32(&c[3*len+4], c31_vreg);
}

float32_t gemm_cm_dot_unroll8x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(const float32_t * __restrict__ a, const float32_t * __restrict__ b, float32_t * __restrict__ c, const uint32_t len) {
    for (uint32_t j = 0; j < len; j += 4) { // j = n
        for (uint32_t i = 0; i < len; i += 8) { // i = m
            addDot8x4_unroll_fused_accumulate_pointers_v2_rowfuse_intrinsics(len, &a[i], &b[j * len], &c[j * len + i], len);
        }
    }
    return c[0];
}
