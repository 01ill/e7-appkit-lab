.syntax unified
.text
.type dotp_scalar, %function
.global dotp_scalar
/*
r0: A -> s1
r1: B -> s2
r2: c -> s0 (Return Value)
r3: Length

r4: Zwischenspeicher für LR
*/

dotp_scalar:
    push {r4, r5, r6, lr} // save r4,r5,r6 and lr
    mov r4, #0
    vdup.32 q0, r4 // init c
    wls lr, r3, loopEnd // while (r3 > 0)

loopStart:
    ldr r5, [r0], #4 // load next A[i]
    vmov.f32 s1, r5
    ldr r6, [r1], #4 // load next B[i]
    vmov.f32 s2, r6
    vmla.f32 s0, s1, s2 // c += A[i] * B[i]
    le lr, loopStart

loopEnd:
    vstr.32 s0, [r2] // save c
    pop {r4, r5, r6, pc} // restore r4,r5,r6 and return
