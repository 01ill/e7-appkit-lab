.syntax unified
.text

.global flops_scalar_fp32
.type flops_scalar_fp32, %function
flops_scalar_fp32:
    push {lr} // save lr
    dls lr, r0 // start loop

flops_scalar_fp32_loop: // 4 * 4 * 2 = 32 FLOP
    vfma.f32 s0, s2, s1
    vfma.f32 s1, s3, s2
    vfma.f32 s2, s0, s3
    vfma.f32 s3, s1, s0

    vfma.f32 s0, s2, s1
    vfma.f32 s1, s3, s2
    vfma.f32 s2, s0, s3
    vfma.f32 s3, s1, s0

    vfma.f32 s0, s2, s1
    vfma.f32 s1, s3, s2
    vfma.f32 s2, s0, s3
    vfma.f32 s3, s1, s0

    vfma.f32 s0, s2, s1
    vfma.f32 s1, s3, s2
    vfma.f32 s2, s0, s3
    vfma.f32 s3, s1, s0


    le lr, flops_scalar_fp32_loop // check loop

flops_scalar_fp32_loop_end:
    pop {pc} // return


.global flops_scalar_fp64
.type flops_scalar_fp64, %function
flops_scalar_fp64:
    push {lr} // save lr
    dls lr, r0 // start loop

flops_scalar_fp64_loop: // 4 * 4 * 2 FLOP = 32
    vfma.f64 d0, d2, d1
    vfma.f64 d1, d3, d2
    vfma.f64 d2, d0, d3
    vfma.f64 d3, d1, d0

    vfma.f64 d0, d2, d1
    vfma.f64 d1, d3, d2
    vfma.f64 d2, d0, d3
    vfma.f64 d3, d1, d0

    vfma.f64 d0, d2, d1
    vfma.f64 d1, d3, d2
    vfma.f64 d2, d0, d3
    vfma.f64 d3, d1, d0

    vfma.f64 d0, d2, d1
    vfma.f64 d1, d3, d2
    vfma.f64 d2, d0, d3
    vfma.f64 d3, d1, d0


    le lr, flops_scalar_fp64_loop // check loop

flops_scalar_fp64_loop_end:
    pop {pc} // return


/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0->r4: scalar
 */
.global flops_mve_fp16
.type flops_mve_fp16, %function
flops_mve_fp16:
    push {lr} // save lr
    dls lr, r0 // start loop

flops_mve_fp16_loop: // 2 * 16 FLOP = 32
    vfma.f16 q0, q2, q1
    vfma.f16 q1, q3, q2

    le lr, flops_mve_fp16_loop // check loop

flops_mve_fp16_loop_end:
    pop {pc} // return

.global flops_mve_fp32
.type flops_mve_fp32, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0->r4: scalar
 */
flops_mve_fp32:
    push {lr} // save lr
    dls lr, r0 // start loop

flops_mve_fp32_loop: // 4*8 = 32 per iteration
    vfma.f32 q0, q2, q1
    vfma.f32 q1, q3, q2
    vfma.f32 q2, q0, q3
    vfma.f32 q3, q1, q0

    le lr, flops_mve_fp32_loop // check loop

flops_mve_fp32_loop_end:
    pop {pc} // return


.global flops_mve_fp32_interleaved
.type flops_mve_fp32_interleaved, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0->r4: scalar
 */
flops_mve_fp32_interleaved:
    push {r4, lr} // save lr
    
    dlstp.32 lr, r3 // start loop

flops_mve_fp32_interleaved_loop:

    .rept OPERATIONAL_INTENSITY // 100 * 8 * 4 = 3200
    vfma.f32 q0, q2, q1
    vldrw.f32 q1, [r2]
    vfma.f32 q1, q3, q2
    vldrw.f32 q1, [r2]
    vfma.f32 q2, q0, q3
    vldrw.f32 q0, [r1]
    vfma.f32 q3, q1, q0
    vldrw.f32 q0, [r1]
    .endr

    letp lr, flops_mve_fp32_interleaved_loop // check loop

flops_mve_fp32_interleaved_loop_end:
    pop {r4, pc} // return


.global flops_mve_fp32_vec4
.type flops_mve_fp32_vec4, %function
/*
 * r0: *a
 * r1: *b
 * r2: *c
 * r3: len
 * s0->r4: scalar
 */
flops_mve_fp32_vec4:
    push {lr} // save lr
    vpush {q4-q7}


    dls lr, r0 // start loop


flops_mve_fp32_vec4_loop:
    vfma.f32 q4, q0, q1
    vfma.f32 q5, q0, q1
    vfma.f32 q4, q0, q1
    vfma.f32 q5, q0, q1

    vfma.f32 q6, q2, q3
    vfma.f32 q7, q2, q3
    vfma.f32 q6, q2, q3
    vfma.f32 q7, q2, q3

    le lr, flops_mve_fp32_vec4_loop // check loop

flops_mve_fp32_vec4_loop_end:
    vpop {q4-q7}
    pop {pc} // return

