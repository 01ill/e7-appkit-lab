.syntax unified

.text
.global gemm_asm_4x4
.type gemm_asm_4x4, %function

/*
r0: *a
r1: *b
r2: *c
r3: len

q0-q3: C-Accumulator Register
q4: A-Register

*/
gemm_asm_4x4:
    push {r4-r12, lr}
    vpush {q4} // save q4

    mov r10, r0 // store base a
    mov r11, r1 // store base b
    mov r12, r2 // store base c

    mov r4, #0 // j counter
    //push {r10, r11, r12}

gemm_loop_j:
    cmp r4, r3
    bge gemm_loop_j_end
    mov r5, #0 // i counter

gemm_loop_i:
    //pop {r10, r11, r12}
    cmp r5, r3
    bge gemm_loop_i_end

    /* Tile Pointer berechnen */
    // a[i]
    add r0, r10, r5, lsl #2 // a + i*4
    // b[j * len]
    mul r2, r4, r3
    add r1, r11, r2, lsl #2 // b + j*len*4
    // c[j * len + i]
    add r2, r12, r2, lsl #2 // c + j*len*4
    add r2, r2, r5, lsl #2 // c + i*4
    //push {r10, r11, r12}
    /* Init Accumulator Block */
    vmov.i32 q0, #0.0
    vorr.f32 q1, q0, q0
    vorr.f32 q2, q0, q0
    vorr.f32 q3, q0, q0

    add r7, r1, r3, lsl #3 // b[2len]
    add r6, r1, r3, lsl #2 // b[len]
    add r8, r7, r3, lsl #2 // b[3len]
    push {r10, r11, r12}
    wls lr, r3, gemm_asm_4x4_end // loop over k; jump to end if no elements left

gemm_asm_4x4_loop: // 32flops * 16 
    /* Load A */
    vldrw.f32 q4, [r0]
    add r0, r0, r3, lsl #2 // Einmal Länge aufaddieren

    /*ldr r9, [r7], #4 // load b[2len]
    ldr r10, [r8], #4
    ldr r11, [r6], #4 // load b[len]
    ldr r12, [r1], #4 // load b[0]*/
    vldrw.f32 q7, [r1], #16
    vfma.f32 q2, q4, q7
    vfma.f32 q3, q4, q7
    vfma.f32 q1, q4, q7
    vfma.f32 q0, q4, q7

    le lr, gemm_asm_4x4_loop

gemm_asm_4x4_end:
    pop {r10, r11, r12}
    vstrw.f32 q0, [r2] // store c[0]
    vstrw.f32 q1, [r2] // store c[len]
    vstrw.f32 q2, [r2] // store c[2len]
    vstrw.f32 q3, [r2] // store c[3len]

    add r5, r5, #4
    b gemm_loop_i

gemm_loop_i_end:
    add r4, r4, #4
    b gemm_loop_j

gemm_loop_j_end:
    vpop {q4}
    pop {r4-r12, pc}
