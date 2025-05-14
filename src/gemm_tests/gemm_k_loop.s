.global gemm_asm_konly
.type gemm_asm_konly, %function

.equ C00_OFFSET, (0)
.equ C01_OFFSET, (16)
.equ C10_OFFSET, (24*4)
.equ C11_OFFSET, (24*4+16)
.equ C20_OFFSET, (48*4)
.equ C21_OFFSET, (48*4+16)
.equ NEXT_A, (23*96-8*4)

/*
r0: *a
r1: *b
r2: *c
r3: len

q0-q5: C-Accumulator Register
q0 q1
q2 q3
q4 q5

q6, q7: A-Register
q6 q7

r4: j counter
r5: i counter
r6: bp0 pointer + temp 0 for c accumulator block
r7: bp1 pointer
r8: bp2 pointer
r9: bp3 pointer

r10-r12: base for a-c

*/
gemm_asm_konly:
    push {r4-r9, lr}
    vpush {q4-q7} // save vector registers (we need all of them)


    /* Init Accumulator Block */
    //vdup.32 q0, r6 // Initialize Accumulator Block
    // mul r9, r3, #6 // k loop wird 6*k ausgeführt. Damit wird nur ein Pointer auf b genutzt
    dls lr, r3 // loop over k; jump to end if no elements left

/*
Operational Intensity:
- Pro Schleifendurchlauf:
- 8 Elemente A (256Bit=32 Byte)
- 3 Elemente B (96 Bit = 12 Byte)
- 6*VFMA = 8*6 FLOPS = 48 FLOP
--> 48/44 = 1.1 OI
 */
gemm_asm_konly_loop: // 32flops * 16 
    vfma.f32 q2, q6, r7     // [LOOP K] calculate c[1,0]
    ldr r9, [r1], #4        // [LOOP K] load b[0] and write back for next k
    //ldr r8, [r1, #192] // [LOOP K] load b[2len]
    vfma.f32 q4, q6, r8     // [LOOP K] calculate c[2,0]
    vfma.f32 q0, q6, r9     // [LOOP K] calculate c[0,0]
    vldrw.f32 q7, [r0, #16] // [LOOP K] load next a[1]
    vfma.f32 q1, q7, r9     // [LOOP K] calculate c[0,1]
    add.w r0, r0, r3, lsl #2 // Einmal Länge aufaddieren (next k)
    vfma.f32 q3, q7, r7     // [LOOP K] calculate c[1,1]
    vldrw.f32 q6, [r0]      // [LOOP K NEXT] load next a[0]
    vfma.f32 q5, q7, r8     // [LOOP K] calculate c[2,1]

    ldr r7, [r1, r3, lsl #2] // [LOOP K NEXT] load b[len]
    //ldr r8, [r1, r3] // [LOOP K] load b[2len]
    ldr r8, [r1, r3, lsl #3] // [LOOP K PRE] load b[2len]

    le lr, gemm_asm_konly_loop

gemm_loop_konly_end:
    vpop {q4-q7}
    pop {r4-r9, pc}

.global gemm_asm_kloop_microkernel
.type gemm_asm_kloop_microkernel, %function

/*
r0: *a
r1: *b
r2: *c
r3: len

q0-q5: C-Accumulator Register
q0 q1
q2 q3
q4 q5

q6, q7: A-Register
q6 q7

r4: j counter
r5: i counter
r6: bp0 pointer + temp 0 for c accumulator block
r7: bp1 pointer
r8: bp2 pointer
r9: bp3 pointer

r10-r12: base for a-c

*/
gemm_asm_kloop_microkernel:
    push {r4-r6, lr}
    vpush {q4-q7} // save vector registers (we need all of them)

    dls lr, r3 // loop over k; jump to end if no elements left

/*
Operational Intensity:
- Pro Schleifendurchlauf:
- 8 Elemente A (256Bit=32 Byte)
- 3 Elemente B (96 Bit = 12 Byte)
- 6*VFMA = 8*6 FLOPS = 48 FLOP
--> 48/44 = 1.1 OI
 */
 .p2align 2
gemm_asm_kloop_microkernel_loop: // 32flops * 16 
    /* Load A */
    ldr.w r4, [r1, r3, lsl #2] // load b[len]
    vldrw.f32 q6, [r0]

    /* Load B */
    ldr.w r5, [r1, r3, lsl #3] // load b[2len]
    vfma.f32 q2, q6, r4
    ldr.w r6, [r1], #4 // load b[0] and write back for next k
    vfma.f32 q4, q6, r5
    vldrw.f32 q7, [r0, #16]
    vfma.f32 q0, q6, r6
    vfma.f32 q3, q7, r4
    vfma.f32 q5, q7, r5
    add.w r0, r0, r3, lsl #2 // Einmal Länge aufaddieren (next k)
    vfma.f32 q1, q6, r6

    le lr, gemm_asm_kloop_microkernel_loop

gemm_asm_kloop_microkernel_loop_end:
    vpop {q4-q7}
    pop {r4-r6, lr}
