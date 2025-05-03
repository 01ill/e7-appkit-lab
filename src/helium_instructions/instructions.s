.text

.macro SIZE, label
    .size label, (. - label)
.endm

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