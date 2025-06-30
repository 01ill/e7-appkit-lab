.syntax unified
.text

.global throughput_mve
.type throughput_mve, %function

throughput_mve:
    push {lr}

    dlstp.32 lr, r1
.p2align 2
throughput_mve_loop:
    vldrw.f32 q0, [r0], #16
    letp lr, throughput_mve_loop

throughput_mve_loop_end:
    pop {pc}