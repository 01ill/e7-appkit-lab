.syntax unified

.text
.global gemm_asm_4x7
.type gemm_asm_4x7, %function

/*
r0: *a
r1: *b
r2: *c
r3: len

q0-q5: C-Accumulator Register
q6: A-Register

r4: j counter
r5: i counter
r6: bp0 pointer + temp 0 for c accumulator block
r7: bp1 pointer
r8: bp2 pointer
r9: bp3 pointer

r10-r12: base for a-c

*/
gemm_asm_4x7:
    push {r4-r12, lr}
    vpush {q4, q5, q6, q7} // save q4

    mov r10, r0 // store base a
    mov r11, r1 // store base b
    mov r12, r2 // store base c

    mov r4, #0 // j counter


gemm_loop_j:
    cmp r4, r3
    bge gemm_loop_j_end
    @ sub r4, r4, #0x4 // next tile
    @ cbz r4, gemm_loop_j_end
    mov r5, #0 // i counter

gemm_loop_i:
    cmp r5, r3
    bge gemm_loop_i_end
    @ sub r5, r5, #0x4 // next tile
    @ cbz r5, gemm_loop_i_end

    /* Tile Pointer berechnen */
    // a[i]
    add r0, r10, r5, lsl #2 // a + i*4
    // b[j * len]
    mul r2, r4, r3
    add r1, r11, r2, lsl #2 // b + j*len*4
    // c[j * len + i]
    add r2, r12, r2, lsl #2 // c + j*len*4
    add r2, r2, r5, lsl #2 // c + i*4

    /* Init Accumulator Block */
    mov r6, #0.0
    vdup.32 q0, r6 // Initialize Accumulator Block
    vdup.32 q1, r6
    vdup.32 q2, r6
    vdup.32 q3, r6
    vdup.32 q4, r6
    vdup.32 q5, r6
    vdup.32 q7, r6

_micro_kernel_start:
    wls lr, r3, gemm_asm_4x6_end // loop over k; jump to end if no elements left

gemm_asm_4x6_loop: // 32flops * 16 
    /* Load A */
    vldrw.f32 q6, [r0]
    add r0, r0, r3, lsl #2 // Einmal Länge aufaddieren

    ldr.w r7, [r1, r3, lsl #3] // load b[2len]
    vfma.f32 q2, q6, r7

    // ldr r8, [r7, r3, lsl #2] // load b[3len]
    vfma.f32 q3, q6, r8

    // load b[4len]
    ldr.w r7, [r1, r3, lsl #2]
    vfma.f32 q4, q6, r7

    // load b[5len]
    ldr.w r8, [r1, r3, lsl #2]
    vfma.f32 q5, q6, r8

    ldr.w r7, [r1, r3, lsl #2] // load b[len]
    vfma.f32 q1, q6, r7

    ldr.w r7, [r1, r3, lsl #2] // load b[len]
    vfma.f32 q7, q6, r7

    ldr.w r8, [r1], #4 // load b[0] and rewrite for new k
    vfma.f32 q0, q6, r8

    le lr, gemm_asm_4x6_loop

gemm_asm_4x6_end:
    vstrw.f32 q0, [r2] // store c[0]
    vstrw.f32 q1, [r2] // store c[len]
    vstrw.f32 q2, [r2] // store c[2len]
    vstrw.f32 q3, [r2] // store c[3len]
    vstrw.f32 q4, [r2]
    vstrw.f32 q5, [r2]
    vstrw.f32 q7, [r2]

    add r5, r5, #4
    b gemm_loop_i

gemm_loop_i_end:
    add r4, r4, #6
    b gemm_loop_j

gemm_loop_j_end:
    vpop {q4, q5, q6, q7}
    pop {r4-r12, pc}
