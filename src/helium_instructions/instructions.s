.text


.type dotp_mve, %function
.global dotp_mve


.macro SIZE, label
    .size label, (. - label)
.endm