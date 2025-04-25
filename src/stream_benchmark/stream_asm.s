.syntax unified
.text

/* --- COPY --- */
.global stream_copy
.type stream_copy, %function

/*
 * r0: *a
 * r1: *c
 * r2: len
 */
stream_copy:
    push {lr} // save lr
    wls lr, r2, stream_copy_end // start loop

stream_copy_loop:
    ldr r3, [r0], #4 // load from a
    str r3, [r1], #4 // copy to c
    le lr, stream_copy_loop // check loop

stream_copy_end:
    pop {pc} // return


.global stream_copy_mve
.type stream_copy_mve, %function
/*
 * r0: *a
 * r1: *c
 * r2: len
 */
stream_copy_mve:
    push {lr} // save lr
    wlstp.32 lr, r2, stream_copy_mve_end // start loop

stream_copy_mve_loop:
    vldrw.f32 q0, [r0], #16 // load 4 elements from a
    vstrw.f32 q0, [r1], #16 // copy to c

    letp lr, stream_copy_mve_loop // check loop

stream_copy_mve_end:
    pop {pc} // return

/* --- SCALE --- */
.global stream_scale
.type stream_scale, %function
/*
 * r0: *c
 * r1: *b
 * s0: scalar (Floats werden in Float-Register übergeben)
 * r2: len
 */
stream_scale:
    push {lr} // save lr
    dls lr, r2 // start loop

stream_scale_loop:
    ldr r3, [r0], #4 // load from c
    vmov.f32 s1, r3 // move c[i] to s1
    vmul.f32 s2, s1, s0 // scale
    vstr.32 s2, [r1] // copy to b
    add r1, r1, #4 // next b
    le lr, stream_scale_loop // check loop

stream_scale_end:
    pop {pc} // return

.global stream_scale_mve
.type stream_scale_mve, %function
/*
 * r0: *c
 * r1: *b
 * s0: scalar
 * r2: len
 */
stream_scale_mve:
    vmov.f32 r3, s0 // scalar to r3
    push {lr} // save lr
    dlstp.32 lr, r2 // start loop

stream_scale_mve_loop:
    vldrw.f32 q0, [r0], #16 // load 4 elements from c
    vmul.f32 q1, q0, r3 // scale
    vstrw.f32 q1, [r1], #16 // copy to b
    letp lr, stream_scale_mve_loop // check loop

stream_scale_mve_end:
    pop {pc} // return

/* --- ADD --- */
.global stream_add
.type stream_add, %function
/*
 * r0: *c
 * r1: *a
 * r2: *b
 * r3: len
 */
stream_add:
    push {r4, r5, lr} // save lr
    wls lr, r3, stream_add_end // start loop

stream_add_loop:
    ldr r4, [r1], #4 // load from a
    vmov.f32 s0, r4 // move a[i] to s0
    ldr r5, [r2], #4 // load from b
    vmov.f32 s1, r5 // move b[i] to s1
    vadd.f32 s2, s1, s0 // add a[i] + b[i]
    vstr.32 s2, [r0] // copy to c
    add r0, r0, #4 // next c
    le lr, stream_add_loop // check loop

stream_add_end:
    pop {r4, r5, pc} // return

.global stream_add_mve
.type stream_add_mve, %function
/*
 * r0: *c
 * r1: *a
 * r2: *b
 * r3: len
 */
stream_add_mve:
    push {lr} // save lr
    wlstp.32 lr, r3, stream_add_mve_end // start loop

stream_add_mve_loop:
    vldrw.f32 q0, [r1], #16 // load 4 elements from a
    vldrw.f32 q1, [r2], #16 // load 4 elements from b
    vadd.f32 q2, q0, q1 // add a[i] + b[i]  
    vstrw.f32 q2, [r0], #16 // copy to c
    letp lr, stream_add_mve_loop // check loop

stream_add_mve_end:
    pop {pc} // return

/* --- TRIAD --- */
.global stream_triad
.type stream_triad, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0: scalar
 */
stream_triad:
    push {r4, r5, lr} // save lr
    wls lr, r3, stream_triad_end // start loop

stream_triad_loop:
    ldr r5, [r2], #4 // load from c
    vmov.f32 s2, r5 // move c[i] to s2
    vmul.f32 s2, s2, s0 // scale c[i]

    ldr r4, [r1], #4 // load from b
    vmov.f32 s1, r4 // move b[i] to s1
    vadd.f32 s2, s2, s0 // add b[i] + scaled c[i]
    vstr.32 s2, [r0] // copy to a
    add r0, r0, #4 // next a
    le lr, stream_triad_loop // check loop

stream_triad_end:
    pop {r4, r5, pc} // return

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
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    wlstp.32 lr, r3, stream_triad_mve_end // start loop

stream_triad_mve_loop:
    vldrw.f32 q1, [r2], #16 // load 4 elements from c
    vldrw.f32 q0, [r1], #16 // load 4 elements from b
    vfma.f32 q0, q1, r4 // b[i] + scalar * c[i] = b[i]
    vstrw.f32 q0, [r0], #16 // copy to a
    letp lr, stream_triad_mve_loop // check loop

stream_triad_mve_end:
    pop {r4, pc} // return

