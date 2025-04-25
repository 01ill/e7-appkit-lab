.syntax unified

.text
.type gemm_4x6, %function
.global gemm_4x6


.macro RANK1_UPDATE_4x6
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
gemm_4x6:

    push {r4, lr} // save r4 and lr

    mov r4, #0x0 // base offset for strides 
    vidup.u32 q7, r2, #4 // setup strides (6 columns of float32 (4 bytes))
    vadd.s32 q7, q7, r2 // add base address to strides
    vldrw.f32 q0, [q1] // load first column of c (gather load)
    vldrw.f32 q0, [q1, #16]! // load second column of c (gather load)
    vldrw.f32 q0, [q1, #16]! // load third column of c (gather load)
    vldrw.f32 q0, [q1, #16]! // load fourth column of c (gather load)
    vldrw.f32 q0, [q1, #16]! // load fifth column of c (gather load)
    vldrw.f32 q0, [q1] // load sixth column of c (gather load)

    // load c


    vldrw.f32 q0, [r2], #16 // first column
    vldrw.f32 q1, [r2], #16 // second column
    vldrw.f32 q2, [r2], #16 // third column
    vldrw.f32 q3, [r2], #16 // fourth column
    vldrw.f32 q4, [r2], #16 // fifth column
    vldrw.f32 q5, [r2] // sixth column

    wls lr, r3, end

loopRank1:
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


    le lr, loopRank1 // check loop

end:
    // store c
    vstrw.f32 q5, [r2], #-16 // sixth column
    vstrw.f32 q4, [r2], #-16 // fifth column
    vstrw.f32 q3, [r2], #-16 // fourth column
    vstrw.f32 q2, [r2], #-16 // third column
    vstrw.f32 q1, [r2], #-16 // second column
    vstrw.f32 q0, [r2], #-16 // first column


    pop {r4, pc} // restore r4 and return
