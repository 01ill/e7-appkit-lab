.global gemm_asm_8x3
.type gemm_asm_8x3, %function

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
gemm_asm_8x3:
    push {r4-r12, lr}
    vpush {q4-q7} // save vector registers (we need all of them)

    mov r10, r0 // store base a
    mov r11, r1 // store base b
    mov r12, r2 // store base c

    mov r4, #0 // j counter
    //mov r4, r3 // j counter

gemm_loop_j:
    // cmp r4, r3
    // bge gemm_loop_j_end
    @ sub r4, r4, #0x4 // next tile
    @ cbz r4, gemm_loop_j_end
    //mov r5, r3 // i counter
    mov r5, #0 // i counter
    sub r6, r3, #2

gemm_loop_i:
    // cmp r5, r3
    // bge gemm_loop_i_end
    @ sub r5, r5, #0x4 // next tile
    @ cbz r5, gemm_loop_i_end

    /* Tile Pointer berechnen */
    // a[i]
    /*add r0, r10, r5, lsl #2 // a + i*4
    // b[j * len]
    mul r2, r4, r3
    add r1, r11, r2, lsl #2 // b + j*len*4
    // c[j * len + i]
    add r2, r12, r2, lsl #2 // c + j*len*4
    add r2, r2, r5, lsl #2 // c + i*4*/

    ldr.w r7, [r1, r3, lsl #2] // load b[len]
    ldr.w r8, [r1, r3, lsl #3] // load b[2len]
    ldr.w r9, [r1], #4 // load b[0] and write back for next k
    vmov.i32 q5, #0         // [INIT ACCUMULATOR] c[2,1]
    vldrw.f32 q6, [r0]      // [LOOP K PRE] load next A[0]
    vorr.f32 q0, q5, q5     // [INIT ACCUMULATOR] c[0,0]
    vfma.f32 q0, q6, r9     // [LOOP K PRE] calculate c[0,0]
    vorr.f32 q1, q5, q5         // [INIT ACCUMULATOR] c[0,1]
    vldrw.f32 q7, [r0, #16] // [LOOP K PRE] load next A[1]
    vfma.f32 q1, q7, r9     // [LOOP K PRE] calculate c[0,1]
    vorr.f32 q2, q5, q5         // [INIT ACCUMULATOR] c[1,0]
    vfma.f32 q2, q6, r7     // [LOOP K PRE] calculate c[1,0]
    vorr.f32 q3, q5, q5         // [INIT ACCUMULATOR] c[1,1]
    vfma.f32 q3, q7, r7     // [LOOP K PRE] calculate c[1,1]
    vorr.f32 q4, q5, q5         // [INIT ACCUMULATOR] c[2,0]
    vfma.f32 q4, q6, r8     // [LOOP K PRE] calculate c[2,0]
    //add.w r0, r0, r3, lsl #2 // Einmal Länge aufaddieren (next k)
    vldrw.f32 q6, [r0, #24*4]  // [LOOP K BEFORE] load next A[0]
    vfma.f32 q5, q7, r8     // [LOOP K PRE] calculate c[2,1]
    
    add.w r0, r0, r3, lsl #2
    ldr r7, [r1, #96]// [LOOP K BEFORE] load b[len]
    //ldr r8, [r1, r3] // [LOOP K] load b[2len]
    ldr r8, [r1, #192] // [LOOP K PRE] load b[2len]

    /* Init Accumulator Block */
    //vdup.32 q0, r6 // Initialize Accumulator Block
    // mul r9, r3, #6 // k loop wird 6*k ausgeführt. Damit wird nur ein Pointer auf b genutzt
    dls lr, r6 // loop over k; jump to end if no elements left

/*
Operational Intensity:
- Pro Schleifendurchlauf:
- 8 Elemente A (256Bit=32 Byte)
- 3 Elemente B (96 Bit = 12 Byte)
- 6*VFMA = 8*6 FLOPS = 48 FLOP
--> 48/44 = 1.1 OI
 */
gemm_asm_8x3_loop: // 32flops * 16 
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

    le lr, gemm_asm_8x3_loop

gemm_asm_8x3_end:
    ldr.w r9, [r1], #4 // load b[0] and write back for next k
    vldrw.f32 q7, [r0, #16]
    vfma.f32 q1, q7, r9
    vstrw.f32 q1, [r2, #16] // store c[4:7]
    vfma.f32 q0, q6, r9
    vstrw.f32 q0, [r2] // store c[0:3]
    vfma.f32 q2, q6, r7
    vstrw.f32 q2, [r2, #C10_OFFSET] // store c[2len]
    vfma.f32 q3, q7, r7
    vstrw.f32 q3, [r2, #C11_OFFSET] // store c[2len+1]‚
    vfma.f32 q4, q6, r8
    vstrw.f32 q4, [r2, #C20_OFFSET] // store c[3len]
    vfma.f32 q5, q7, r8
    vstrw.f32 q5, [r2, #C21_OFFSET] // store c[5len]

    // Alternative for calculating tile pointers: rewind pointers
    // A:
    // 
    add r5, r5, #8

    add r0, r10, r5, lsl #2 // rewind a
    sub r1, r1, #4*24
    add r2, r2, #8*4


    cmp r5, r3
    blt gemm_loop_i
    //subs r5, r5, #8
    //bge gemm_loop_i
    # b gemm_loop_i

gemm_loop_i_end:
    mov r0, r10
    add r1, r1, #24*3*4
    add r2, r2, #2*24*4

    add r4, r4, #3
    cmp r4, r3
    blt gemm_loop_j
    //subs r4, r4, #3
    //bge gemm_loop_j
    // b gemm_loop_j

gemm_loop_j_end:
    vpop {q4-q7}
    pop {r4-r12, pc}


.global gemm_asm_8x3_microkernel
.type gemm_asm_8x3_microkernel, %function

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
gemm_asm_8x3_microkernel:
    push {r4-r6, lr}
    vpush {q4-q7} // save vector registers (we need all of them)

    mov r4, #0 // j counter

    mov r6, #0.0
    vmov.i32 q0, #0.0
    vorr.f32 q1, q0, q0
    vdup.32 q2, r6
    vorr.f32 q3, q0, q0
    vdup.32 q4, r6
    vorr.f32 q5, q0, q0

    wls lr, r3, gemm_asm_8x3_microkernel_loop_end // loop over k; jump to end if no elements left

/*
Operational Intensity:
- Pro Schleifendurchlauf:
- 8 Elemente A (256Bit=32 Byte)
- 3 Elemente B (96 Bit = 12 Byte)
- 6*VFMA = 8*6 FLOPS = 48 FLOP
--> 48/44 = 1.1 OI
 */
 .p2align 2
gemm_asm_8x3_microkernel_loop: // 32flops * 16 
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

    le lr, gemm_asm_8x3_microkernel_loop

gemm_asm_8x3_microkernel_loop_end:
    vstrw.f32 q0, [r2] // store c[0:3]
    vstrw.f32 q1, [r2, #16] // store c[4:7]
    add r2, r2, r3, lsl #2
    vstrw.f32 q2, [r2] // store c[2len]
    vstrw.f32 q3, [r2, #16] // store c[2len+1]
    add r2, r2, r3, lsl #2
    vstrw.f32 q4, [r2] // store c[3len]
    vstrw.f32 q5, [r2, #16] // store c[5len]

    vpop {q4-q7}
    pop {r4-r6, lr}
