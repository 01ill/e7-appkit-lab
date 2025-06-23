.text

.macro SIZE, label
    .size label, (. - label)
.endm

.p2align 3

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