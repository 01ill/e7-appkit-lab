.syntax unified
.text
.p2align 2

.global throughput_mve_read
.type throughput_mve_read, %function

throughput_mve_read:
    push.w {lr}

    dlstp.32 lr, r1
throughput_mve_loop_read:
    vldrw.f32 q0, [r0], #16
    letp lr, throughput_mve_loop_read

throughput_mve_loop_end_read:
    pop.w {pc}

.global throughput_mve_read2
.type throughput_mve_read2, %function

throughput_mve_read2:
    push.w {lr}

    dlstp.16 lr, r2
throughput_mve_loop_read2:
    vldrw.f32 q0, [r0], #16
    vldrw.f32 q1, [r1], #16
    letp lr, throughput_mve_loop_read2

throughput_mve_loop_end_read2:
    pop.w {pc}


.global throughput_scalar_read
.type throughput_scalar_read, %function

throughput_scalar_read:
    push.w {lr}

    dls lr, r1
throughput_scalar_read_loop:
    ldr r2, [r0], #4
    le lr, throughput_scalar_read_loop

throughput_scalar_read_loop_end:
    pop.w {pc}

.global throughput_scalar_write
.type throughput_scalar_write, %function

throughput_scalar_write:
    push.w {lr}

    dls lr, r1
throughput_scalar_write_loop:
    ldr r2, [r0], #4
    le lr, throughput_scalar_write_loop

throughput_scalar_write_loop_end:
    pop.w {pc}

.global throughput_mve_write
.type throughput_mve_write, %function

throughput_mve_write:
    push.w {lr}

    dlstp.32 lr, r1
throughput_mve_loop_write:
    vstrw.f32 q0, [r0], #16
    letp lr, throughput_mve_loop_write

throughput_mve_loop_end_write:
    pop.w {pc}