; =============================================================================
; NanoGEMM — High-Performance AVX2+FMA Microkernels (x86-64 FASM)
; =============================================================================

; -----------------------------------------------------------------------------
; void nanogemm_matmul(int M, int N, int K, const float* A, const float* B, float* C)
; Windows x64 ABI:
; RCX = M, RDX = N, R8 = K, R9 = A, [RSP+40] = B, [RSP+48] = C
; -----------------------------------------------------------------------------
nanogemm_matmul:
    push rbp
    mov rbp, rsp

    ; Save caller register parameters into shadow space
    mov [rbp + 16], rcx     ; M
    mov [rbp + 24], rdx     ; N
    mov [rbp + 32], r8      ; K
    mov [rbp + 40], r9      ; A

    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 280            ; local frame + XMM saves

    ; Save non-volatile XMM registers (xmm6 - xmm15)
    movups [rsp + 0],   xmm6
    movups [rsp + 16],  xmm7
    movups [rsp + 32],  xmm8
    movups [rsp + 48],  xmm9
    movups [rsp + 64],  xmm10
    movups [rsp + 80],  xmm11
    movups [rsp + 96],  xmm12
    movups [rsp + 112], xmm13
    movups [rsp + 128], xmm14
    movups [rsp + 144], xmm15

    ; Early exit on non-positive dimensions
    mov eax, [rbp + 16]     ; M
    test eax, eax
    jle .matmul_done
    mov eax, [rbp + 24]     ; N
    test eax, eax
    jle .matmul_done
    mov eax, [rbp + 32]     ; K
    test eax, eax
    jle .matmul_done

    ; Precompute strides in bytes
    ; [rsp + 160] = stride_B = N * 4
    ; [rsp + 168] = stride_C = N * 4
    ; [rsp + 176] = stride_A = K * 4
    movsxd rax, dword [rbp + 24]    ; N
    shl rax, 2
    mov [rsp + 160], rax
    mov [rsp + 168], rax

    movsxd rax, dword [rbp + 32]    ; K
    shl rax, 2
    mov [rsp + 176], rax

    ; K loop limit
    movsxd r10, dword [rbp + 32]

    ; Outer row loop (steps of 4)
    xor esi, esi                    ; i = 0

.row_loop_4:
    lea eax, [esi + 4]
    cmp eax, dword [rbp + 16]       ; i + 4 <= M?
    jg .row_tail_prep

    ; Calculate row pointers for A: row_r_A = A + (i + r) * K * 4
    mov rax, rsi
    imul rax, [rsp + 176]           ; i * stride_A
    mov r8, [rbp + 40]              ; A
    add r8, rax
    mov [rsp + 184], r8             ; row 0
    add r8, [rsp + 176]
    mov [rsp + 192], r8             ; row 1
    add r8, [rsp + 176]
    mov [rsp + 200], r8             ; row 2
    add r8, [rsp + 176]
    mov [rsp + 208], r8             ; row 3

    ; Calculate row pointers for C: row_r_C = C + (i + r) * N * 4
    mov rax, rsi
    imul rax, [rsp + 168]           ; i * stride_C
    mov r8, [rbp + 56]              ; C
    add r8, rax
    mov [rsp + 216], r8             ; row 0
    add r8, [rsp + 168]
    mov [rsp + 224], r8             ; row 1
    add r8, [rsp + 168]
    mov [rsp + 232], r8             ; row 2
    add r8, [rsp + 168]
    mov [rsp + 240], r8             ; row 3

    ; Column loop (steps of 16)
    xor edi, edi                    ; j = 0

.col_loop_16:
    lea eax, [edi + 16]
    cmp eax, dword [rbp + 24]       ; j + 16 <= N?
    jg .col_loop_8

    ; Initialize 8 accumulators to 0
    vxorps ymm4,  ymm4,  ymm4
    vxorps ymm5,  ymm5,  ymm5
    vxorps ymm6,  ymm6,  ymm6
    vxorps ymm7,  ymm7,  ymm7
    vxorps ymm8,  ymm8,  ymm8
    vxorps ymm9,  ymm9,  ymm9
    vxorps ymm10, ymm10, ymm10
    vxorps ymm11, ymm11, ymm11

    ; B_ptr = B + j * 4
    mov r11, [rbp + 48]             ; B
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx                    ; k = 0

.k_loop_16:
    vmovups ymm0, [r11]
    vmovups ymm1, [r11 + 32]

    ; Row 0
    mov rax, [rsp + 184]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm4, ymm2, ymm0
    vfmadd231ps ymm5, ymm2, ymm1

    ; Row 1
    mov rax, [rsp + 192]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm6, ymm2, ymm0
    vfmadd231ps ymm7, ymm2, ymm1

    ; Row 2
    mov rax, [rsp + 200]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm8, ymm2, ymm0
    vfmadd231ps ymm9, ymm2, ymm1

    ; Row 3
    mov rax, [rsp + 208]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm10, ymm2, ymm0
    vfmadd231ps ymm11, ymm2, ymm1

    add r11, [rsp + 160]            ; B_ptr += stride_B
    inc rbx
    cmp rbx, r10
    jl .k_loop_16

    ; Store results to C
    mov rax, [rsp + 216]
    vmovups [rax + rdi*4],      ymm4
    vmovups [rax + rdi*4 + 32], ymm5

    mov rax, [rsp + 224]
    vmovups [rax + rdi*4],      ymm6
    vmovups [rax + rdi*4 + 32], ymm7

    mov rax, [rsp + 232]
    vmovups [rax + rdi*4],      ymm8
    vmovups [rax + rdi*4 + 32], ymm9

    mov rax, [rsp + 240]
    vmovups [rax + rdi*4],      ymm10
    vmovups [rax + rdi*4 + 32], ymm11

    add edi, 16
    jmp .col_loop_16

.col_loop_8:
    lea eax, [edi + 8]
    cmp eax, dword [rbp + 24]       ; j + 8 <= N?
    jg .col_scalar_prep

    vxorps ymm4,  ymm4,  ymm4
    vxorps ymm6,  ymm6,  ymm6
    vxorps ymm8,  ymm8,  ymm8
    vxorps ymm10, ymm10, ymm10

    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx

.k_loop_8:
    vmovups ymm0, [r11]

    mov rax, [rsp + 184]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm4, ymm2, ymm0

    mov rax, [rsp + 192]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm6, ymm2, ymm0

    mov rax, [rsp + 200]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm8, ymm2, ymm0

    mov rax, [rsp + 208]
    vbroadcastss ymm2, [rax + rbx*4]
    vfmadd231ps ymm10, ymm2, ymm0

    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_loop_8

    mov rax, [rsp + 216]
    vmovups [rax + rdi*4], ymm4
    mov rax, [rsp + 224]
    vmovups [rax + rdi*4], ymm6
    mov rax, [rsp + 232]
    vmovups [rax + rdi*4], ymm8
    mov rax, [rsp + 240]
    vmovups [rax + rdi*4], ymm10

    add edi, 8

.col_scalar_prep:
    cmp edi, dword [rbp + 24]       ; j >= N?
    jge .row_4_next

.col_scalar_loop:
    ; Row 0
    vxorps xmm0, xmm0, xmm0
    mov r8,  [rsp + 184]
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx
.k_sc_0:
    vmovss xmm1, [r8 + rbx*4]
    vmulss xmm1, xmm1, [r11]
    vaddss xmm0, xmm0, xmm1
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_sc_0
    mov rax, [rsp + 216]
    vmovss [rax + rdi*4], xmm0

    ; Row 1
    vxorps xmm0, xmm0, xmm0
    mov r8,  [rsp + 192]
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx
.k_sc_1:
    vmovss xmm1, [r8 + rbx*4]
    vmulss xmm1, xmm1, [r11]
    vaddss xmm0, xmm0, xmm1
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_sc_1
    mov rax, [rsp + 224]
    vmovss [rax + rdi*4], xmm0

    ; Row 2
    vxorps xmm0, xmm0, xmm0
    mov r8,  [rsp + 200]
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx
.k_sc_2:
    vmovss xmm1, [r8 + rbx*4]
    vmulss xmm1, xmm1, [r11]
    vaddss xmm0, xmm0, xmm1
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_sc_2
    mov rax, [rsp + 232]
    vmovss [rax + rdi*4], xmm0

    ; Row 3
    vxorps xmm0, xmm0, xmm0
    mov r8,  [rsp + 208]
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx
.k_sc_3:
    vmovss xmm1, [r8 + rbx*4]
    vmulss xmm1, xmm1, [r11]
    vaddss xmm0, xmm0, xmm1
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_sc_3
    mov rax, [rsp + 240]
    vmovss [rax + rdi*4], xmm0

    inc edi
    cmp edi, dword [rbp + 24]
    jl .col_scalar_loop

.row_4_next:
    add esi, 4
    jmp .row_loop_4

.row_tail_prep:
    cmp esi, dword [rbp + 16]       ; i >= M?
    jge .matmul_done

.row_tail_loop:
    mov rax, rsi
    imul rax, [rsp + 176]
    mov r8, [rbp + 40]
    add r8, rax
    mov [rsp + 184], r8             ; row_i_A

    mov rax, rsi
    imul rax, [rsp + 168]
    mov r8, [rbp + 56]
    add r8, rax
    mov [rsp + 216], r8             ; row_i_C

    xor edi, edi

.row_tail_col_8:
    lea eax, [edi + 8]
    cmp eax, dword [rbp + 24]
    jg .row_tail_col_sc

    vxorps ymm4, ymm4, ymm4
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    mov r8, [rsp + 184]
    xor ebx, ebx

.k_tail_8:
    vmovups ymm0, [r11]
    vbroadcastss ymm2, [r8 + rbx*4]
    vfmadd231ps ymm4, ymm2, ymm0
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_tail_8

    mov rax, [rsp + 216]
    vmovups [rax + rdi*4], ymm4
    add edi, 8
    jmp .row_tail_col_8

.row_tail_col_sc:
    cmp edi, dword [rbp + 24]
    jge .row_tail_next

.row_tail_sc_loop:
    vxorps xmm0, xmm0, xmm0
    mov r8,  [rsp + 184]
    mov r11, [rbp + 48]
    mov rax, rdi
    shl rax, 2
    add r11, rax
    xor ebx, ebx

.k_tail_sc:
    vmovss xmm1, [r8 + rbx*4]
    vmulss xmm1, xmm1, [r11]
    vaddss xmm0, xmm0, xmm1
    add r11, [rsp + 160]
    inc rbx
    cmp rbx, r10
    jl .k_tail_sc

    mov rax, [rsp + 216]
    vmovss [rax + rdi*4], xmm0
    inc edi
    cmp edi, dword [rbp + 24]
    jl .row_tail_sc_loop

.row_tail_next:
    inc esi
    cmp esi, dword [rbp + 16]
    jl .row_tail_loop

.matmul_done:
    vzeroupper
    movups xmm6,  [rsp + 0]
    movups xmm7,  [rsp + 16]
    movups xmm8,  [rsp + 32]
    movups xmm9,  [rsp + 48]
    movups xmm10, [rsp + 64]
    movups xmm11, [rsp + 80]
    movups xmm12, [rsp + 96]
    movups xmm13, [rsp + 112]
    movups xmm14, [rsp + 128]
    movups xmm15, [rsp + 144]
    add rsp, 280
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret

; -----------------------------------------------------------------------------
; void nanogemm_sgemm(int M, int N, int K, float alpha,
;                     const float* A, int lda,
;                     const float* B, int ldb,
;                     float beta,
;                     float* C, int ldc)
; -----------------------------------------------------------------------------
nanogemm_sgemm:
    push rbp
    mov rbp, rsp

    ; Save register arguments into shadow space
    mov [rbp + 16], rcx     ; M
    mov [rbp + 24], rdx     ; N
    mov [rbp + 32], r8      ; K
    movss [rbp + 40], xmm3  ; alpha in shadow space [rbp+40]

    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 136

    movups [rsp + 0],  xmm6
    movups [rsp + 16], xmm7

    mov eax, [rbp + 16]     ; M
    test eax, eax
    jle .sgemm_done
    mov eax, [rbp + 24]     ; N
    test eax, eax
    jle .sgemm_done
    mov eax, [rbp + 32]     ; K
    test eax, eax
    jle .sgemm_done

    vbroadcastss ymm6, [rbp + 40]   ; valpha in ymm6
    vbroadcastss ymm7, [rbp + 80]   ; vbeta in ymm7

    ; Loop over rows i = 0 .. M-1
    xor esi, esi            ; i = 0

.sgemm_i_loop:
    movsxd rax, dword [rbp + 56]    ; lda
    imul rax, rsi
    shl rax, 2
    mov r8, [rbp + 48]              ; A
    add r8, rax                     ; row_i_A in r8

    movsxd rax, dword [rbp + 96]    ; ldc
    imul rax, rsi
    shl rax, 2
    mov r9, [rbp + 88]              ; C
    add r9, rax                     ; row_i_C in r9

    xor edi, edi            ; j = 0

.sgemm_j_8:
    lea eax, [edi + 8]
    cmp eax, dword [rbp + 24]
    jg .sgemm_j_sc

    vxorps ymm4, ymm4, ymm4
    mov r11, [rbp + 64]             ; B
    mov rax, rdi
    shl rax, 2
    add r11, rax                    ; &B[0, j]
    movsxd r12, dword [rbp + 72]    ; ldb
    shl r12, 2                      ; ldb * 4
    xor ebx, ebx                    ; k = 0
    movsxd r10, dword [rbp + 32]    ; K

.sgemm_k_8:
    vmovups ymm0, [r11]
    vbroadcastss ymm2, [r8 + rbx*4]
    vfmadd231ps ymm4, ymm2, ymm0
    add r11, r12                    ; advance B by ldb * 4
    inc rbx
    cmp rbx, r10
    jl .sgemm_k_8

    ; Scale sum by alpha: ymm4 = ymm4 * alpha
    vmulps ymm4, ymm4, ymm6

    ; If beta != 0: ymm4 = ymm4 + beta * C
    movss xmm0, [rbp + 80]          ; beta
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    je .sgemm_store_8

    vmovups ymm0, [r9 + rdi*4]
    vfmadd231ps ymm4, ymm0, ymm7

.sgemm_store_8:
    vmovups [r9 + rdi*4], ymm4
    add edi, 8
    jmp .sgemm_j_8

.sgemm_j_sc:
    cmp edi, dword [rbp + 24]
    jge .sgemm_i_next

.sgemm_j_sc_loop:
    vxorps xmm4, xmm4, xmm4
    mov r11, [rbp + 64]             ; B
    mov rax, rdi
    shl rax, 2
    add r11, rax                    ; &B[0, j]
    movsxd r12, dword [rbp + 72]    ; ldb
    shl r12, 2
    xor ebx, ebx
    movsxd r10, dword [rbp + 32]    ; K

.sgemm_k_sc:
    vmovss xmm0, [r8 + rbx*4]
    vmulss xmm0, xmm0, [r11]
    vaddss xmm4, xmm4, xmm0
    add r11, r12
    inc rbx
    cmp rbx, r10
    jl .sgemm_k_sc

    ; Scale sum by alpha
    vmulss xmm4, xmm4, [rbp + 40]

    ; If beta != 0
    movss xmm0, [rbp + 80]          ; beta
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    je .sgemm_store_sc

    vmovss xmm1, [r9 + rdi*4]
    vmulss xmm1, xmm1, xmm0
    vaddss xmm4, xmm4, xmm1

.sgemm_store_sc:
    vmovss [r9 + rdi*4], xmm4
    inc edi
    cmp edi, dword [rbp + 24]
    jl .sgemm_j_sc_loop

.sgemm_i_next:
    inc esi
    cmp esi, dword [rbp + 16]
    jl .sgemm_i_loop

.sgemm_done:
    vzeroupper
    movups xmm6, [rsp + 0]
    movups xmm7, [rsp + 16]
    add rsp, 136
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret
