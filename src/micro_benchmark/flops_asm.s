.syntax unified
.text


/*
Operational Intensity: FLOP/Byte
For Triad two vector registers are loaded, i.e. 2*128bit = 256 Bit = 32 Byte
This means, to reach an operational intensity of 1, we need 32 FLOP.
One (32Bit) VFMA instruction gets us 8 FLOPS so we need 4 VFMA instructions.

One 32 Bit 
*/


.global flops_scalar_fp32
.type flops_scalar_fp32, %function
flops_scalar_fp32:
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4

    vldrw.f32 q1, [r2], #16 // load 4 elements from c
    vldrw.f32 q0, [r1], #16 // load 4 elements from b

    wlstp.32 lr, r3, flops_scalar_fp32_loop_end // start loop

flops_scalar_fp32_loop:
    .rept OPERATIONAL_INTENSITY // 100 * 2 * 4 = 800
    vfma.f32 s0, s2, s1
    vfma.f32 s1, s3, s2
    vfma.f32 s2, s0, s3
    vfma.f32 s3, s1, s0
    .endr

    letp lr, flops_scalar_fp32_loop // check loop

flops_scalar_fp32_loop_end:
    vstrw.f32 q0, [r0], #16 // copy to a

    pop {r4, pc} // return


.global flops_scalar_fp64
.type flops_scalar_fp64, %function
flops_scalar_fp64:
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    wlstp.32 lr, r3, flops_scalar_fp64_loop_end // start loop
    vldrw.f32 q1, [r2], #16 // load 4 elements from c
    vldrw.f32 q0, [r1], #16 // load 4 elements from b

flops_scalar_fp64_loop:
    .rept OPERATIONAL_INTENSITY // 100 * 2 * 4 = 800
    vfma.f64 d0, d2, d1
    vfma.f64 d1, d3, d2
    vfma.f64 d2, d0, d3
    vfma.f64 d3, d1, d0
    .endr

    letp lr, flops_scalar_fp64_loop // check loop

flops_scalar_fp64_loop_end:
    vstrw.f32 q0, [r0], #16 // copy to a
    pop {r4, pc} // return


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
    push {r4, lr} // save lr
    vmov.f16 r4, s0 // scalar to r4


    wlstp.16 lr, r3, flops_mve_fp16_loop_end // start loop


flops_mve_fp16_loop:
    vldrh.f16 q1, [r2] // load 8 elements from c
    vldrh.f16 q0, [r1] // load 8 elements from b

    .rept OPERATIONAL_INTENSITY // 100 * 16 * 4 = 6400
    vfma.f16 q0, q2, q1
    vfma.f16 q1, q3, q2
    vfma.f16 q2, q0, q3
    vfma.f16 q3, q1, q0
    .endr

    vstrh.f16 q0, [r0] // copy to a
    letp lr, flops_mve_fp16_loop // check loop

flops_mve_fp16_loop_end:

    pop {r4, pc} // return


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
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4

    wlstp.32 lr, r3, flops_mve_fp32_loop_end // start loop

flops_mve_fp32_loop:
    vldrw.f32 q1, [r2]
    vldrw.f32 q0, [r1]

    .rept OPERATIONAL_INTENSITY // 100 * 8 * 4 = 3200
    vfma.f32 q0, q2, q1
    vfma.f32 q1, q3, q2
    vfma.f32 q2, q0, q3
    vfma.f32 q3, q1, q0
    .endr

    vstrw.f32 q0, [r0] // copy to a


    letp lr, flops_mve_fp32_loop // check loop

flops_mve_fp32_loop_end:
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
    push {r4, lr} // save lr
    vmov.f32 r4, s0 // scalar to r4
    vpush {q4-q7}

    vldrw.f32 q0, [r1], #16
    vldrw.f32 q2, [r1], #16
    vldrw.f32 q2, [r2], #16
    vldrw.f32 q3, [r2], #16


    wlstp.16 lr, r3, flops_mve_fp32_vec4_loop_end // start loop


flops_mve_fp32_vec4_loop:
    vldrw.f32 q0, [r1], #16
    vldrw.f32 q2, [r1], #16

    .rept OPERATIONAL_INTENSITY // 100 * 8 * 4 = 3200
    vfma.f32 q4, q0, q1
    vfma.f32 q5, q0, q1
    vfma.f32 q4, q0, q1
    vfma.f32 q5, q0, q1

    vfma.f32 q6, q2, q3
    vfma.f32 q7, q2, q3
    vfma.f32 q6, q2, q3
    vfma.f32 q7, q2, q3
    .endr

    vldrw.f32 q2, [r2], #16
    vldrw.f32 q3, [r2], #16


    // vstrw.f32 q0, [r0] // copy to a


    letp lr, flops_mve_fp32_vec4_loop // check loop

flops_mve_fp32_vec4_loop_end:
    vpop {q4-q7}
    pop {r4, pc} // return

