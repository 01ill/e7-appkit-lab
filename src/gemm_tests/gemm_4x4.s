.syntax unified

.text
.type gemm_4x4, %function
.global gemm_4x4


.macro RANK1_UPDATE_4x4
    vldrw.f32 q6, [r0], #16 // next column of A
    
    ldr r4, [r1], #4    // load B[0]
    vfma.f32 q0, q6, r4 // update first column of C
    
    ldr r4, [r1], #4    // load B[1]
    vfma.f32 q1, q6, r4 // update second column of C

    ldr r4, [r1], #4    // load B[2]
    vfma.f32 q2, q6, r4 // update third column of C

    ldr r4, [r1], #4    // load B[3]
    vfma.f32 q3, q6, r4 // update fourth column of C

    ldr r4, [r1], #4    // load B[4]
    vfma.f32 q4, q6, r4 // update fifth column of C

    ldr r4, [r1], #4    // load B[5]
    vfma.f32 q5, q6, r4 // update sixth column of C
.endm

/*
r0: *A
r1: *B
r2: *C
r3: k
r4: Skalar Wert von B

q0-q5: C
q6: Column of A
*/
gemm_4x4:
    push {r4, r5, lr} // save r4 and lr

    vld40.32 {q0, q1, q2, q3}, [r2] // interleaved first row
    vld41.32 {q0, q1, q2, q3}, [r2] // interleaved second row
    vld42.32 {q0, q1, q2, q3}, [r2] // interleaved third row
    vld43.32 {q0, q1, q2, q3}, [r2] // interleaved fourth row

    // vidup.u32 q4, r5, #4 

    // q7 ... s28, s29, s30, s31
    mov r4, #0x0
    mov r5, #0x4
    vidup.u32 q7, r4, #4 // setup stride (4 bytes)
    vmul.u32 q7, q7, r3 // multiply by k -> offsets for first column of A (0, 4, 8, 12)
    wls lr, r3, end

loopRank1:
    vldrw.f32 q6, [r0, q7] // load column of A
//    vldrw.f32 q6, [r0], #16 // next column of A
    
    ldr r4, [r1], #4    // load B[0]
    vfma.f32 q0, q6, r4 // update first column of C
    
    ldr r4, [r1], #4    // load B[1]
    vfma.f32 q1, q6, r4 // update second column of C

    ldr r4, [r1], #4    // load B[2]
    vfma.f32 q2, q6, r4 // update third column of C

    ldr r4, [r1], #4    // load B[3]
    vfma.f32 q3, q6, r4 // update fourth column of C

    vadd.u32 q7, q7, r5 // increment offset for next column of A

    le lr, loopRank1 // check loop

end:
    // store c
    vst40.32 {q0, q1, q2, q3}, [r2] // interleaved first row
    vst41.32 {q0, q1, q2, q3}, [r2] // interleaved second row
    vst42.32 {q0, q1, q2, q3}, [r2] // interleaved third row
    vst43.32 {q0, q1, q2, q3}, [r2] // interleaved fourth row


    pop {r4, r5, pc} // restore r4 and return
