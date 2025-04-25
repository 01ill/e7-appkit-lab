.text
.type dotp_mve, %function
.global dotp_mve

/*
r0 + q0: A, r1 + q1: B, r2 + q2: C

-- Am Ende: s0 = c[0], s1 = c[1], s2 = c[2], s3 = c[3]

r3: Länge des Arrays
r4: LR Zwischenspeicher
*/
dotp_mve:
    push {r4, r5, lr} // save r4,r5 and lr
    mov r5, #0
    vdup.32 q2, r5 // init c
    dlstp.32 lr, r3 // start loop

loopMVE:
    vldrw.f32 q0, [r0], #16 // Vier Elemente von A laden
    vldrw.f32 q1, [r1], #16 // Vier Elemente von B laden

    vfma.f32 q2, q0, q1 // c[i] += A[i] * B[i]

    letp lr, loopMVE // check loop

loopEnd:
    // q2[0] = s8, q2[1] = s9, q2[2] = s10, q2[3] = s11
    // sum up c
    vadd.f32 s0, s8, s9
    vadd.f32 s0, s0, s10
    vadd.f32 s0, s0, s11

    // vstrw.f32 q2, [r2] // c speichern
    vstr.32 s0, [r2]
    
    // mov lr, r4 // restore lr
    pop {r4, r5, pc} // restore r4,r5 and return
