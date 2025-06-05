.syntax unified

.text
.type gemm_20x24_jit, %function
.global gemm_20x24_jit

gemm_20x24_jit:
    stmdb   sp!, {r4, r5, r6, r7, r8, r9, sl, fp, ip, lr}
    vpush   {d8-d15}
    mov.w   r3, r0
    movw    r4, #0
gemm_jit_j_start:
    movw    r5, #0
gemm_jit_i_start:
    nop.w
    movw    r6, #4
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #0]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vldrw.u32       q7, [r0, #16]
    ldr.w   r7, [r1], #4
    vldrw.u32       q1, [r2, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q2, [r2, #80]
    vfma.f32        q2, q6, r8
    vldrw.u32       q3, [r2, #96]
    vfma.f32        q3, q7, r8
    vldrw.u32       q4, [r2, #160]
    vfma.f32        q4, q6, r9
    vldrw.u32       q5, [r2, #176]
    vfma.f32        q5, q7, r9
    vldrw.u32       q0, [r2, #0]
    vfma.f32        q0, q6, r7
    vldrw.u32       q6, [r0, #80]
    addw    r0, r0, #80     @ 0x50
    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    dls     lr, r6
gemm_jit_k_start:
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #80]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #96]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #160]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #176]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #240]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #256]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #320]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #336]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #400]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #400    @ 0x190
    le      lr, gemm_jit_k_start

    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #80]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #96]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #160]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #160    @ 0xa0
    ldr.w   r7, [r1], #4
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vstrw.32        q1, [r2, #16]
    vfma.f32        q0, q6, r7
    vstrw.32        q0, [r2, #0]
    vfma.f32        q2, q6, r8
    vstrw.32        q2, [r2, #80]
    vfma.f32        q3, q7, r8
    vstrw.32        q3, [r2, #96]
    vfma.f32        q5, q7, r9
    vstrw.32        q5, [r2, #176]
    vfma.f32        q4, q6, r9
    vstrw.32        q4, [r2, #160]
    addw    r5, r5, #8
    add.w   r0, r3, r5, lsl #2
    subw    r1, r1, #96     @ 0x60
    addw    r2, r2, #32
    cmp.w   r5, #16
    blt.w   gemm_jit_i_start
    movw    r6, #4
    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    vldrw.u32       q6, [r0, #0]
    ldr.w   r7, [r1], #4
    vldrw.u32       q0, [r2, #0]
    vfma.f32        q0, q6, r7
    vldrw.u32       q2, [r2, #80]
    vfma.f32        q2, q6, r8
    vldrw.u32       q4, [r2, #160]
    vfma.f32        q4, q6, r9
    vldrw.u32       q6, [r0, #80]
    addw    r0, r0, #80     @ 0x50
    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    dls     lr, r6
gemm_jit_k_tail_start:
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #80]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #160]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #240]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #320]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #400]
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #400    @ 0x190
    le      lr, gemm_jit_k_tail_start
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #80]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #160]
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #160    @ 0xa0
    ldr.w   r7, [r1], #4
    vfma.f32        q0, q6, r7
    vstrw.32        q0, [r2, #0]
    vfma.f32        q2, q6, r8
    vstrw.32        q2, [r2, #80]
    vfma.f32        q4, q6, r9
    vstrw.32        q4, [r2, #160]
    subw    r1, r1, #96     @ 0x60
    addw    r2, r2, #16
    mov.w   r0, r3
    addw    r1, r1, #288    @ 0x120
    addw    r2, r2, #160    @ 0xa0
    addw    r4, r4, #3
    cmp.w   r4, #24
    blt.w   gemm_jit_j_start
    vpop    {d8-d15}
    ldmia.w sp!, {r4, r5, r6, r7, r8, r9, sl, fp, ip, pc}

.type gemm_20x24_tuned, %function
.global gemm_20x24_tuned

gemm_20x24_tuned:
    stmdb   sp!, {r4, r5, r6, r7, r8, r9, sl, fp, ip, lr}
    vpush   {d8-d15}
    mov.w   r3, r0
    movw    r4, #0
gemm_tuned_j_start:
    movw    r5, #0
gemm_tuned_i_start:
    movw    r6, #4
    ldr.w   r8, [r1, #96]   @ 0x60
    vldrw.u32       q6, [r0, #0]
    ldr.w   r9, [r1, #192]  @ 0xc0
    vldrw.u32       q7, [r0, #16]
    ldr.w   r7, [r1], #4
    vldrw.u32       q1, [r2, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q2, [r2, #80]
    vfma.f32        q2, q6, r8
    vldrw.u32       q3, [r2, #96]
    vfma.f32        q3, q7, r8
    vldrw.u32       q4, [r2, #160]
    vfma.f32        q4, q6, r9
    vldrw.u32       q5, [r2, #176]
    vfma.f32        q5, q7, r9
    vldrw.u32       q0, [r2, #0]
    vfma.f32        q0, q6, r7
    vldrw.u32       q6, [r0, #80]
    addw    r0, r0, #80     @ 0x50
    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    dls     lr, r6
gemm_tuned_k_start:
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #80]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #96]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #160]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #176]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #240]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #256]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #320]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #336]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #400]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #400    @ 0x190
    le      lr, gemm_tuned_k_start

    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #80]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    vfma.f32        q0, q6, r7
    vldrw.u32       q7, [r0, #96]
    vfma.f32        q1, q7, r7
    vldrw.u32       q6, [r0, #160]
    vfma.f32        q3, q7, r8
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32        q5, q7, r9
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #160    @ 0xa0
    ldr.w   r7, [r1], #4
    vldrw.u32       q7, [r0, #16]
    vfma.f32        q1, q7, r7
    vstrw.32        q1, [r2, #16]
    vfma.f32        q0, q6, r7
    vstrw.32        q0, [r2, #0]
    vfma.f32        q2, q6, r8
    vstrw.32        q2, [r2, #80]
    vfma.f32        q3, q7, r8
    vstrw.32        q3, [r2, #96]
    vfma.f32        q5, q7, r9
    vstrw.32        q5, [r2, #176]
    vfma.f32        q4, q6, r9
    vstrw.32        q4, [r2, #160]
    addw    r5, r5, #8
    add.w   r0, r3, r5, lsl #2
    subw    r1, r1, #96     @ 0x60
    addw    r2, r2, #32
    cmp.w   r5, #16
    blt.w   gemm_tuned_i_start

    and r6, r4, #1 // check if uneven
    cmp r6, #1
    beq.w gemm_tuned_tail_end
    # addw    r6, r4, #3
    # cmp.w   r6, #24
    # bge.w gemm_tuned_tail_end

    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    ldr.w   r7, [r1], #4
    vldrw.u32       q6, [r0, #0]
    movw    r6, #4
    vldrw.u32       q0, [r2, #0]
    vfma.f32        q0, q6, r7
    vldrw.u32       q2, [r2, #80]
    vfma.f32        q2, q6, r8
    vldrw.u32       q4, [r2, #160]
    vfma.f32        q4, q6, r9
    
    ldr.w   r7, [r1, #3*24*4-4]
    ldr.w   r8, [r1, #4*24*4-4]
    ldr.w   r9, [r1, #5*24*4-4]

    vldrw.u32       q1, [r2, #3*20*4]
    vfma.f32        q1, q6, r7
    vldrw.f32       q3, [r2, #4*20*4]
    vfma.f32        q3, q6, r8
    vldrw.u32       q5, [r2, #5*20*4]
    vfma.f32        q5, q6, r9
    
    vldrw.u32       q6, [r0, #80]
    addw    r0, r0, #80     @ 0x50
    ldr.w   r8, [r1, #96]   @ 0x60
    ldr.w   r9, [r1, #192]  @ 0xc0
    dls     lr, r6
gemm_tuned_k_tail_start:
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #80]
    ldr.w   r9, [r1, #192]  @ 0xc0

    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #160]
    ldr.w   r9, [r1, #192]  @ 0xc0
    
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #240]
    ldr.w   r9, [r1, #192]  @ 0xc0
    
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #320]
    ldr.w   r9, [r1, #192]  @ 0xc0
    
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #400]
    ldr.w   r9, [r1, #192]  @ 0xc0
    addw    r0, r0, #400    @ 0x190
    le      lr, gemm_tuned_k_tail_start

    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #80]
    ldr.w   r9, [r1, #192]  @ 0xc0
    
    vfma.f32        q2, q6, r8
    ldr.w   r7, [r1], #4
    vfma.f32        q4, q6, r9
    ldr.w   r8, [r1, #4*24*4-4]
    vfma.f32        q0, q6, r7
    ldr.w r7, [r1, #3*24*4-4]
    vfma.f32    q3, q6, r8
    ldr.w r9, [r1, #5*24*4-4]
    vfma.f32    q1, q6, r7
    ldr.w   r8, [r1, #96]   @ 0x60
    vfma.f32    q5, q6, r9
    vldrw.u32       q6, [r0, #160]
    ldr.w   r9, [r1, #192]  @ 0xc0
    
    ldr.w   r7, [r1], #4
    vfma.f32        q0, q6, r7
    vstrw.32        q0, [r2, #0]
    vfma.f32        q2, q6, r8
    vstrw.32        q2, [r2, #80]
    vfma.f32        q4, q6, r9
    vstrw.32        q4, [r2, #160]
    ldr.w           r7, [r1, #3*24*4-4]
    ldr.w           r8, [r1, #4*24*4-4]
    ldr.w           r9, [r1, #5*24*4-4]

    vfma.f32 q1, q6, r7
    vstrw.32 q1, [r2, #3*20*4]

    vfma.f32    q3, q6, r8
    vstrw.32    q3, [r2, #4*20*4]

    vfma.f32 q5, q6, r9
    vstrw.f32 q5, [r2, #5*20*4]




    subw    r1, r1, #96     @ 0x60
    // addw    r2, r2, #16
gemm_tuned_tail_end:
    addw r2, r2, #16 // still move c because we already calculated

    mov.w   r0, r3
    addw    r1, r1, #288    @ 0x120
    addw    r2, r2, #160    @ 0xa0
    addw    r4, r4, #3
    cmp.w   r4, #24
    blt.w   gemm_tuned_j_start
    vpop    {d8-d15}
    ldmia.w sp!, {r4, r5, r6, r7, r8, r9, sl, fp, ip, pc}