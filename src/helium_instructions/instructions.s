.text
.syntax unified

.global generateFpStall
.type generateFpStall, %function

generateFpStall:
    push {lr}
    // vldrw.f32 q0, [r0]
    ldr r1, [r0], #4
    ldr r2, [r0], #4
    vfma.f32 q0, q1, r2
    vfma.f32 q2, q3, q0

    pop {pc}

.global testPredication
.type testPredication, %function

testPredication:
    push {lr}
    mov r1, #2
    vctp.32 r1
    vpstttt
    vldrwt.f32 q0, [r0]
    vldrwt.f32 q1, [r0, #16]
    vmult.f32 q2, q0, q1
    vstrwt.f32 q2, [r0]
    pop {pc}

.global branch1
.type branch1, %function

branch1:
    push {lr}
    mov r1, #0

loopStartB1:
    vfma.f32 q0, q1, q2
    vfma.f32 q1, q2, q3
    add r1, #1
    vfma.f32 q2, q3, q4
    cmp r1, r0
    vfma.f32 q3, q4, q5

    blt loopStartB1

    pop {pc}

.global branch2
.type branch2, %function

branch2:
    push {lr}
    mov r1, #0

loopStartB2:
    cmp r1, r0
    bge b2End

    vfma.f32 q0, q1, q2
    vfma.f32 q1, q2, q3
    add r1, #1
    vfma.f32 q2, q3, q4
    vfma.f32 q3, q4, q5

    b loopStartB2
b2End:
    pop {pc}


.global branchLOB
.type branchLOB, %function

branchLOB:
    push {lr}
    
    dls lr, r0

loopStartLOB:
    vfma.f32 q0, q1, q2
    vfma.f32 q1, q2, q3
    vfma.f32 q2, q3, q4
    vfma.f32 q3, q4, q5

    le lr, loopStartLOB

    pop {pc}

.global branchCBZ
.type branchCBZ, %function

branchCBZ:
    push {lr}

cbzStart:
    cbz r0, cbzEnd

    vfma.f32 q0, q1, q2
    vfma.f32 q1, q2, q3
    subs r0, #1
    vfma.f32 q2, q3, q4
    vfma.f32 q3, q4, q5
    b cbzStart

cbzEnd:
    pop {pc}

.global testHeliumAlignmentVFMA
.type testHeliumAlignmentVFMA, %function
.p2align 2
testHeliumAlignmentVFMA:
    push.w {lr}

    dls lr, r0

loopLobVFMA:
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2

    le lr, loopLobVFMA

endLoobVFMA:
    pop.w {pc}

.global testHeliumAlignmentNonAlignedVFMA
.type testHeliumAlignmentNonAlignedVFMA, %function
.p2align 2
testHeliumAlignmentNonAlignedVFMA:
    push.w {lr}
    nop.n
    dls lr, r0

loopLobUAVFMA:
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2
    vfma.f32 q0, q1, q2

    le lr, loopLobUAVFMA

endLoobUAVFMA:
    pop.w {pc}

.global testHeliumAlignment
.type testHeliumAlignment, %function
.p2align 2
testHeliumAlignment:
    push.w {lr}
    dls lr, r0

loopLob:
    vldrw.f32 q3, [r1]
    vfma.f32 q0, q1, q2
    vldrw.f32 q3, [r1]
    vfma.f32 q0, q1, q2
    le lr, loopLob

endLoob:
    pop.w {pc}

/**
Start Aligned:
- First 16, second 16: 1100581
- First 32, second 16: 1100584
- Both 32: 1100580

Start unaligned:
- Both 16: 1150562
- First 32, second 16: 1100586
- Both 32: 1150562

*/

.global testHeliumAlignmentNonAligned
.type testHeliumAlignmentNonAligned, %function
.p2align 2
testHeliumAlignmentNonAligned:
    push.w {lr}
    nop.n
    dls lr, r0

loopLobUA:
    vldrw.f32 q3, [r1]
    vfma.f32 q0, q1, q2
    vldrw.f32 q3, [r1]
    vfma.f32 q0, q1, q2
    le lr, loopLobUA

endLoobUA:
    pop.w {pc}

@ .global testHeliumAlignmentLEUnaligned
@ .type testHeliumAlignmentLEUnaligned, %function
@ .p2align 2
@ testHeliumAlignmentLEUnaligned:
@     push.w {lr}
@     dls lr, r0

@ loopLobUA2:
@     vldrw.f32 q3, [r1]
@     vfma.f32 q0, q1, q2
@     vldrw.f32 q3, [r1]
@     vfma.f32 q0, q1, q2
@     le lr, loopLobUA2

@ endLoobUA2:
@     pop {pc}


.global testDualIssue
.type testDualIssue, %function

testDualIssue:
    push.w {lr} // make sure to not dual issue pop/push

    .rept 200

    /* Both 11 */
    @ movs r4, #0
    @ movs r5, #0

    /* 11 First */
    @ movs r4, #0
    @ movs r5, r1

    /* 01 First */
    @ movs r5, r1
    @ movs r4, #0

    /* 32 Bit Second */
    @ movs r5, #0
    @ movs.w r4, #0

    /* 32 Bit First */
    @ movs.w r5, #0
    @ movs r4, #0

    /* 32 Bit Both */
    @ movs.w r5, #0
    @ movs.w r4, #0

    /* CMN 16 Bit */
    @ cmn r1, r2
    @ cmn r3, r4

    /* CMN 32 Bit */
    @ cmn.w r1, r2
    @ cmn.w r3, r4
    adds r1, #4
    cmp r1, #10
    nop.w // make sure no other overlapping exists
    nop.w // make sure no other overlapping exists

    .endr

    pop.w {pc} // make sure to not dual issue pop/push