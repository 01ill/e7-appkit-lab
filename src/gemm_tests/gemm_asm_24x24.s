.syntax unified
.text

.equ C00_OFFSET, (0)
.equ C01_OFFSET, (16)
.equ C10_OFFSET, (24*4)
.equ C11_OFFSET, (24*4+16)
.equ C20_OFFSET, (48*4)
.equ C21_OFFSET, (48*4+16)
.equ NEXT_A, (23*96-8*4)

.global gemm_asm_24x24
.type gemm_asm_24x24, %function
.p2align 2
/*
Kernel for multiplying two 24x24 matrices with each other.
The following optimizations are applied:
- 8x3 microkernel with 100% vector register usage
- instruction interleaving
- unrolling over m loop
- a little bit unrolling on the k loop
--> same performance as the Intrinsics Kernel
--> but only applicable for 24x24 matrices
-- GP Register Usage
r0: *a
r1: *b
r2: *c
r3: len
r8: j counter (n dimension)
r9: len-2 (k loop counter)
r6: store j*len 
r7: bp1 pointer
r4: bp2 pointer
r5: bp3 pointer
r10-r12: base for a-c
-- Vector Register Usage
q0-q5: C-Accumulator Register
q0 q1
q2 q3
q4 q5
q6, q7: A-Register
q6 q7
*/
gemm_asm_24x24:
    push {r4-r12, lr} // save all gp registers
    vpush {q4-q7} // save vector registers (we need all of them)
    /**
     * the length is subtracted by 2 because the k-loop is unrolled a little bit.
     * the first and last iteration are handled by the preceding or following program block
    */
    sub r9, r3, #2
    //mov r9, #22
    mov r10, r0 // store base a
    mov r11, r1 // store base b
    mov r12, r2 // store base c
    mov r8, #0  // j counter
    // mov r3, #192
gemm_loop_j:
    cmp r8, #24              // Loop Check
    bge gemm_loop_j_end
    mov r3, #24
    mul r6, r8, r3          // [TILE POINTER] j*len
    mov r0, r10             // [TILE POINTER] a[0]
    add r1, r11, r6, lsl #2 // [TILE POINTER] b + j*len*4
    add r2, r12, r6, lsl #2 // [TILE POINTER] c + j*len*4
    mov r3, #96
    //mov r3, #192
    ldr r7, [r1, #96] // [LOOP K PRE] load b[len]
    ldr r4, [r1, r3] // [LOOP K PRE] load b[2len]
    //ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]
    ldr r5, [r1], #4        // [LOOP K PRE] load b[0] and write back for next k
    vmov.i32 q5, #0         // [INIT ACCUMULATOR] c[2,1]
    vldrw.f32 q6, [r0]      // [LOOP K PRE] load next A[0]
    vorr.f32 q0, q5, q5     // [INIT ACCUMULATOR] c[0,0]
    vfma.f32 q0, q6, r5     // [LOOP K PRE] calculate c[0,0]
    vorr.f32 q1, q5, q5         // [INIT ACCUMULATOR] c[0,1]
    vldrw.f32 q7, [r0, #16] // [LOOP K PRE] load next A[1]
    vfma.f32 q1, q7, r5     // [LOOP K PRE] calculate c[0,1]
    vorr.f32 q2, q5, q5         // [INIT ACCUMULATOR] c[1,0]
    vfma.f32 q2, q6, r7     // [LOOP K PRE] calculate c[1,0]
    vorr.f32 q3, q5, q5         // [INIT ACCUMULATOR] c[1,1]
    vfma.f32 q3, q7, r7     // [LOOP K PRE] calculate c[1,1]
    vorr.f32 q4, q5, q5         // [INIT ACCUMULATOR] c[2,0]
    vfma.f32 q4, q6, r4     // [LOOP K PRE] calculate c[2,0]
    vldrw.f32 q6, [r0, #24*4]  // [LOOP K BEFORE] load next A[0]
    vfma.f32 q5, q7, r4     // [LOOP K PRE] calculate c[2,1]

    adds r0, #96  // [LOOP K PRE] add length to a pointer (prepare for next k)
    ldr r7, [r1, #96]// [LOOP K BEFORE] load b[len]
    // ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]
    dls lr, r9 // loop over k; jump to end if no elements left

gemm_asm_8x3_loop_0: // 32flops * 16
    vfma.f32 q2, q6, r7     // [LOOP K] calculate c[1,0]
    ldr r5, [r1], #4        // [LOOP K] load b[0] and write back for next k
    //ldr r4, [r1, #192]    // [LOOP K] load b[2len]
    vfma.f32 q4, q6, r4     // [LOOP K] calculate c[2,0]
    vfma.f32 q0, q6, r5     // [LOOP K] calculate c[0,0]
    vldrw.f32 q7, [r0, #16] // [LOOP K] load next a[1]
    vfma.f32 q1, q7, r5     // [LOOP K] calculate c[0,1]
    adds r0, #96            // [LOOP K] Einmal Länge aufaddieren (next k)
    vfma.f32 q3, q7, r7     // [LOOP K] calculate c[1,1]
    vldrw.f32 q6, [r0]      // [LOOP K NEXT] load next a[0]
    vfma.f32 q5, q7, r4     // [LOOP K] calculate c[2,1]

    ldr r7, [r1, #96] // [LOOP K NEXT] load b[len]
    //ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]

    le lr, gemm_asm_8x3_loop_0

gemm_asm_8x3_1:
    //ldr r4, [r1, #192]         // load b[2len]
    ldr r5, [r1], #4                // load b[0] and write back for next k
    vldrw.f32 q7, [r0, #16]         // [LOOP K END] load next a[1]
    vfma.f32 q2, q6, r7             // [LOOP K END] calculate c[1,0]
    vstrw.f32 q2, [r2, #C10_OFFSET] // [STORE] c[1,0]
    vfma.f32 q4, q6, r4             // [LOOP K END] calculate c[2,0]
    vstrw.f32 q4, [r2, #C20_OFFSET] // [STORE] c[2,0]
    vfma.f32 q0, q6, r5             // [LOOP K END] calculate c[0,0]
    vstrw.f32 q0, [r2, #C00_OFFSET] // [STORE] c[0,0]
    vfma.f32 q1, q7, r5             // [LOOP K END] calculate c[0,1]
    vstrw.f32 q1, [r2, #C01_OFFSET] // [STORE] c[0,1]
    vfma.f32 q3, q7, r7             // [LOOP K END] calculate c[1,1]
    vstrw.f32 q3, [r2, #C11_OFFSET] // [STORE] c[1,1]
    vfma.f32 q5, q7, r4             // [LOOP K END] calculate c[2,1]
    vstrw.f32 q5, [r2, #C21_OFFSET] // [STORE] c[2,1]

    /* K LOOP 1 over -> next m */
    //sub r0, r0, NEXT_A
    add r0, r10, #8*4       // [TILE POINTER] advance a to a[8]
    add r1, r11, r6, lsl #2 // [TILE POINTER] reset b to b[j*len]
    adds r2, r2, #8*4        // [TILE POINTER] advance c pointer to next block

    ldr r7, [r1, #96] // [LOOP K PRE] load b[len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]
    ldr r5, [r1], #4        // [LOOP K PRE] load b[0] and write back for next k
    vmov.i32 q5, #0         // [INIT ACCUMULATOR] c[2,1]
    vldrw.f32 q6, [r0]      // [LOOP K PRE] load next A[0]
    vorr.f32 q0, q5, q5     // [INIT ACCUMULATOR] c[0,0]
    vfma.f32 q0, q6, r5     // [LOOP K PRE] calculate c[0,0]
    vorr.f32 q1, q5, q5         // [INIT ACCUMULATOR] c[0,1]
    vldrw.f32 q7, [r0, #16] // [LOOP K PRE] load next A[1]
    vfma.f32 q1, q7, r5     // [LOOP K PRE] calculate c[0,1]
    vorr.f32 q2, q5, q5         // [INIT ACCUMULATOR] c[1,0]
    vfma.f32 q2, q6, r7     // [LOOP K PRE] calculate c[1,0]
    vorr.f32 q3, q5, q5         // [INIT ACCUMULATOR] c[1,1]
    vfma.f32 q3, q7, r7     // [LOOP K PRE] calculate c[1,1]
    vorr.f32 q4, q5, q5         // [INIT ACCUMULATOR] c[2,0]
    vfma.f32 q4, q6, r4     // [LOOP K PRE] calculate c[2,0]
    vldrw.f32 q6, [r0, #24*4]  // [LOOP K BEFORE] load next A[0]
    vfma.f32 q5, q7, r4     // [LOOP K PRE] calculate c[2,1]

    adds r0, #96  // [LOOP K PRE] add length to a pointer (prepare for next k)
    ldr r7, [r1, #96]// [LOOP K BEFORE] load b[len]
    //ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]
    dls lr, r9 // loop over k; jump to end if no elements left

gemm_asm_8x3_loop_1: // 32flops * 16
    vfma.f32 q2, q6, r7     // [LOOP K] calculate c[1,0]
    ldr r5, [r1], #4        // [LOOP K] load b[0] and write back for next k
    //ldr r4, [r1, #192] // [LOOP K] load b[2len]
    vfma.f32 q4, q6, r4     // [LOOP K] calculate c[2,0]
    vfma.f32 q0, q6, r5     // [LOOP K] calculate c[0,0]
    vldrw.f32 q7, [r0, #16] // [LOOP K] load next a[1]
    vfma.f32 q1, q7, r5     // [LOOP K] calculate c[0,1]
    adds r0, #96  // [LOOP K] Einmal Länge aufaddieren (next k)
    vfma.f32 q3, q7, r7     // [LOOP K] calculate c[1,1]
    vldrw.f32 q6, [r0]      // [LOOP K NEXT] load next a[0]
    vfma.f32 q5, q7, r4     // [LOOP K] calculate c[2,1]

    ldr r7, [r1, #96] // [LOOP K NEXT] load b[len]
    //ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]

    le lr, gemm_asm_8x3_loop_1


gemm_asm_8x3_2:
    //ldr r4, [r1, #192]         // load b[2len]
    ldr r5, [r1], #4                // load b[0] and write back for next k
    vldrw.f32 q7, [r0, #16]         // [LOOP K END] load next a[1]
    vfma.f32 q2, q6, r7             // [LOOP K END] calculate c[1,0]
    vstrw.f32 q2, [r2, #C10_OFFSET] // [STORE] c[1,0]
    vfma.f32 q4, q6, r4             // [LOOP K END] calculate c[2,0]
    vstrw.f32 q4, [r2, #C20_OFFSET] // [STORE] c[2,0]
    vfma.f32 q0, q6, r5             // [LOOP K END] calculate c[0,0]
    vstrw.f32 q0, [r2, #C00_OFFSET] // [STORE] c[0,0]
    vfma.f32 q1, q7, r5             // [LOOP K END] calculate c[0,1]
    vstrw.f32 q1, [r2, #C01_OFFSET] // [STORE] c[0,1]
    vfma.f32 q3, q7, r7             // [LOOP K END] calculate c[1,1]
    vstrw.f32 q3, [r2, #C11_OFFSET] // [STORE] c[1,1]
    vfma.f32 q5, q7, r4             // [LOOP K END] calculate c[2,1]
    vstrw.f32 q5, [r2, #C21_OFFSET] // [STORE] c[2,1]

    /* K LOOP 2 over -> next m */
    //sub r0, r0, NEXT_A
    add r0, r10, #16*4      // [TILE POINTER] advance a to a[16]
    add r1, r11, r6, lsl #2 // [TILE POINTER] reset b to b[j*len]
    add r2, r2, #8*4        // [TILE POINTER] advance c pointer to next block
    ldr r7, [r1, #96] // [LOOP K PRE] load b[len]
    //ldr r4, [r1, r3] // [LOOP K PRE] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]
    ldr r5, [r1], #4        // [LOOP K PRE] load b[0] and write back for next k
    vmov.i32 q5, #0         // [INIT ACCUMULATOR] c[2,1]
    vldrw.f32 q6, [r0]      // [LOOP K PRE] load next A[0]
    vorr.f32 q0, q5, q5     // [INIT ACCUMULATOR] c[0,0]
    vfma.f32 q0, q6, r5     // [LOOP K PRE] calculate c[0,0]
    vorr.f32 q1, q5, q5         // [INIT ACCUMULATOR] c[0,1]
    vldrw.f32 q7, [r0, #16] // [LOOP K PRE] load next A[1]
    vfma.f32 q1, q7, r5     // [LOOP K PRE] calculate c[0,1]
    vorr.f32 q2, q5, q5         // [INIT ACCUMULATOR] c[1,0]
    vfma.f32 q2, q6, r7     // [LOOP K PRE] calculate c[1,0]
    vorr.f32 q3, q5, q5         // [INIT ACCUMULATOR] c[1,1]
    vfma.f32 q3, q7, r7     // [LOOP K PRE] calculate c[1,1]
    vorr.f32 q4, q5, q5         // [INIT ACCUMULATOR] c[2,0]
    vfma.f32 q4, q6, r4     // [LOOP K PRE] calculate c[2,0]
    vldrw.f32 q6, [r0, #24*4]  // [LOOP K BEFORE] load next A[0]
    vfma.f32 q5, q7, r4     // [LOOP K PRE] calculate c[2,1]

    adds r0, #96  // [LOOP K PRE] add length to a pointer (prepare for next k)
    ldr r7, [r1, #96]// [LOOP K BEFORE] load b[len]
    //ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]

    dls lr, r9 // loop over k; jump to end if no elements left

gemm_asm_8x3_loop_2: // 32flops * 16
    vfma.f32 q2, q6, r7     // [LOOP K] calculate c[1,0]
    //ldr r4, [r1, #192] // [LOOP K] load b[2len]
    ldr r5, [r1], #4        // [LOOP K] load b[0] and write back for next k
    vfma.f32 q4, q6, r4     // [LOOP K] calculate c[2,0]
    vfma.f32 q0, q6, r5     // [LOOP K] calculate c[0,0]
    vldrw.f32 q7, [r0, #16] // [LOOP K] load next a[1]
    vfma.f32 q1, q7, r5     // [LOOP K] calculate c[0,1]
    adds r0, #96  // [LOOP K] Einmal Länge aufaddieren (next k)
    vfma.f32 q3, q7, r7     // [LOOP K] calculate c[1,1]
    vldrw.f32 q6, [r0]      // [LOOP K NEXT] load next a[0]
    vfma.f32 q5, q7, r4     // [LOOP K] calculate c[2,1]

    ldr r7, [r1, #96] // [LOOP K NEXT] load b[len]
    //ldr r4, [r1, r3] // [LOOP K] load b[2len]
    ldr r4, [r1, #192] // [LOOP K PRE] load b[2len]

    le lr, gemm_asm_8x3_loop_2

gemm_asm_8x3_3:
    // ldr r4, [r1, #192]         // load b[2len]
    ldr r5, [r1], #4                // load b[0] and write back for next k
    vldrw.f32 q7, [r0, #16]         // [LOOP K END] load next a[1]
    vfma.f32 q2, q6, r7             // [LOOP K END] calculate c[1,0]
    vstrw.f32 q2, [r2, #C10_OFFSET] // [STORE] c[1,0]
    vfma.f32 q4, q6, r4             // [LOOP K END] calculate c[2,0]
    vstrw.f32 q4, [r2, #C20_OFFSET] // [STORE] c[2,0]
    vfma.f32 q0, q6, r5             // [LOOP K END] calculate c[0,0]
    vstrw.f32 q0, [r2, #C00_OFFSET] // [STORE] c[0,0]
    vfma.f32 q1, q7, r5             // [LOOP K END] calculate c[0,1]
    vstrw.f32 q1, [r2, #C01_OFFSET] // [STORE] c[0,1]
    vfma.f32 q3, q7, r7             // [LOOP K END] calculate c[1,1]
    vstrw.f32 q3, [r2, #C11_OFFSET] // [STORE] c[1,1]
    vfma.f32 q5, q7, r4             // [LOOP K END] calculate c[2,1]
    vstrw.f32 q5, [r2, #C21_OFFSET] // [STORE] c[2,1]
    adds r8, #3 // next j
    b gemm_loop_j

gemm_loop_j_end:
    /* Restore saved registers */
    vpop {q4-q7}
    pop {r4-r12, pc}
