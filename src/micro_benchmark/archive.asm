/**
Einfache Version 2 mit Vektorregistern und so durchmischt wie es eben geht
- 81000 Elemente
- O3
- Armclang
==> 1873.6 MFLOPS
 */

.global stream_triad_mve
.type stream_triad_mve, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0: scalar
 */
stream_triad_mve:
    //lsr r3, r3, #1 // divide length by 2 because we use the double number of registers
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    wlstp.32 lr, r3, stream_triad_mve_end // start loop


stream_triad_mve_loop:
    vldrw.f32 q0, [r1], #16 // load 4 elements from b
    vldrw.f32 q1, [r2], #16 // load 4 elements from c

    vfma.f32 q0, q1, r4 // b[i] + scalar * c[i] = b[i]

    vstrw.f32 q0, [r0], #16 // copy to a

    letp lr, stream_triad_mve_loop // check loop

stream_triad_mve_end:
    pop {r4, pc} // return

/**
VLDR versucht zu verschatten -> keine Änderung
- 81000 Elemente
- O3
- Armclang
==> 1873.6 MFLOPS

 */
.global stream_triad_mve
.type stream_triad_mve, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0: scalar
 */
stream_triad_mve:
    //lsr r3, r3, #1 // divide length by 2 because we use the double number of registers
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    wlstp.32 lr, r3, stream_triad_mve_end // start loop
    vldrw.f32 q1, [r2], #16 // load 4 elements from c

stream_triad_mve_loop:
    vldrw.f32 q0, [r1], #16 // load 4 elements from b
    vfma.f32 q0, q1, r4 // b[i] + scalar * c[i] = b[i]
    vldrw.f32 q1, [r2], #16 // load 4 elements from c
    vstrw.f32 q0, [r0], #16 // copy to a

    letp lr, stream_triad_mve_loop // check loop

stream_triad_mve_end:
    pop {r4, pc} // return



/**
Version mit 8 Vektorregistern und gar nicht durchmischt
- 81000 Elemente
- O3
- Armclang
==> 1769.5 MFLOPS
*/

.global stream_triad_mve
.type stream_triad_mve, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0: scalar
 */
stream_triad_mve:
    lsr r3, r3, #1 // divide length by 2 because we use the double number of registers
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    wlstp.16 lr, r3, stream_triad_mve_end // start loop


stream_triad_mve_loop:
    vldrw.f32 q1, [r2], #16 // load 4 elements from c
    vldrw.f32 q0, [r1], #16 // load 4 elements from b
    vldrw.f32 q2, [r2], #16 // load 4 elements from c
    vldrw.f32 q3, [r1], #16 // load 4 elements from b
    vldrw.f32 q4, [r2], #16 // load 4 elements from c
    vldrw.f32 q5, [r1], #16 // load 4 elements from b
    vldrw.f32 q6, [r2], #16 // load 4 elements from c
    vldrw.f32 q7, [r1], #16 // load 4 elements from b


    vfma.f32 q0, q1, r4 // b[i] + scalar * c[i] = b[i]
    vfma.f32 q3, q2, r4 // b[i] + scalar * c[i] = b[i]
    vfma.f32 q5, q4, r4 // b[i] + scalar * c[i] = b[i]
    vfma.f32 q7, q6, r4 // b[i] + scalar * c[i] = b[i]

    vstrw.f32 q0, [r0], #16 // copy to a
    vstrw.f32 q3, [r0], #16 // copy to a
    vstrw.f32 q5, [r0], #16
    vstrw.f32 q7, [r0], #16

    letp lr, stream_triad_mve_loop // check loop

stream_triad_mve_end:
    pop {r4, pc} // return
