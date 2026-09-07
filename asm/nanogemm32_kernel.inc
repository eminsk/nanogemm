; =============================================================================
; NanoGEMM — High-Performance 32-bit SSE2 Matrix Microkernels (x86 FASM)
; =============================================================================

; -----------------------------------------------------------------------------
; const char* nanogemm_simd_isa(void)
; -----------------------------------------------------------------------------
nanogemm_simd_isa:
    lea eax, [isa_str]
    ret

; -----------------------------------------------------------------------------
; void __cdecl nanogemm_matmul(int M, int N, int K, const float* A, const float* B, float* C)
; Stack layout:
; [ebp + 8]  = M
; [ebp + 12] = N
; [ebp + 16] = K
; [ebp + 20] = A
; [ebp + 24] = B
; [ebp + 28] = C
; -----------------------------------------------------------------------------
nanogemm_matmul:
    push ebp
    mov ebp, esp

    push ebx
    push esi
    push edi
    sub esp, 48             ; local variables

    ; Local variables:
    ; [esp + 0]  = stride_B = N * 4
    ; [esp + 4]  = stride_C = N * 4
    ; [esp + 8]  = stride_A = K * 4
    ; [esp + 12] = row0_A
    ; [esp + 16] = row1_A
    ; [esp + 20] = row2_A
    ; [esp + 24] = row3_A
    ; [esp + 28] = row0_C
    ; [esp + 32] = row1_C
    ; [esp + 36] = row2_C
    ; [esp + 40] = row3_C

    ; Early exit on non-positive dimensions
    mov eax, [ebp + 8]      ; M
    test eax, eax
    jle .matmul_done
    mov eax, [ebp + 12]     ; N
    test eax, eax
    jle .matmul_done
    mov eax, [ebp + 16]     ; K
    test eax, eax
    jle .matmul_done

    ; Precompute strides
    mov eax, [ebp + 12]     ; N
    shl eax, 2
    mov [esp + 0], eax      ; stride_B
    mov [esp + 4], eax      ; stride_C

    mov eax, [ebp + 16]     ; K
    shl eax, 2
    mov [esp + 8], eax      ; stride_A

    ; Outer row loop (steps of 4)
    xor esi, esi            ; i = 0

.row_loop_4:
    lea eax, [esi + 4]
    cmp eax, [ebp + 8]      ; i + 4 <= M?
    jg .row_tail_prep

    ; Row pointers for A: row_r_A = A + (i + r) * stride_A
    mov eax, esi
    imul eax, [esp + 8]
    mov edx, [ebp + 20]     ; A
    add edx, eax
    mov [esp + 12], edx     ; row 0
    add edx, [esp + 8]
    mov [esp + 16], edx     ; row 1
    add edx, [esp + 8]
    mov [esp + 20], edx     ; row 2
    add edx, [esp + 8]
    mov [esp + 24], edx     ; row 3

    ; Row pointers for C: row_r_C = C + (i + r) * stride_C
    mov eax, esi
    imul eax, [esp + 4]
    mov edx, [ebp + 28]     ; C
    add edx, eax
    mov [esp + 28], edx     ; row 0
    add edx, [esp + 4]
    mov [esp + 32], edx     ; row 1
    add edx, [esp + 4]
    mov [esp + 36], edx     ; row 2
    add edx, [esp + 4]
    mov [esp + 40], edx     ; row 3

    ; Column loop (steps of 4)
    xor edi, edi            ; j = 0

.col_loop_4:
    lea eax, [edi + 4]
    cmp eax, [ebp + 12]     ; j + 4 <= N?
    jg .col_scalar_prep

    ; Initialize 4 accumulators to 0
    xorps xmm4, xmm4
    xorps xmm5, xmm5
    xorps xmm6, xmm6
    xorps xmm7, xmm7

    ; B_ptr = B + j * 4
    mov edx, [ebp + 24]     ; B
    lea eax, [edi * 4]
    add edx, eax
    xor ebx, ebx            ; k = 0

.k_loop_4:
    movups xmm0, [edx]

    ; Row 0
    mov eax, [esp + 12]
    movss xmm1, [eax + ebx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm4, xmm1

    ; Row 1
    mov eax, [esp + 16]
    movss xmm1, [eax + ebx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm5, xmm1

    ; Row 2
    mov eax, [esp + 20]
    movss xmm1, [eax + ebx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm6, xmm1

    ; Row 3
    mov eax, [esp + 24]
    movss xmm1, [eax + ebx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm7, xmm1

    add edx, [esp + 0]      ; B_ptr += stride_B
    inc ebx
    cmp ebx, [ebp + 16]     ; k < K?
    jl .k_loop_4

    ; Store results to C
    mov eax, [esp + 28]
    movups [eax + edi * 4], xmm4
    mov eax, [esp + 32]
    movups [eax + edi * 4], xmm5
    mov eax, [esp + 36]
    movups [eax + edi * 4], xmm6
    mov eax, [esp + 40]
    movups [eax + edi * 4], xmm7

    add edi, 4
    jmp .col_loop_4

.col_scalar_prep:
    cmp edi, [ebp + 12]     ; j >= N?
    jge .row_4_next

.col_scalar_loop:
    ; Row 0
    xorps xmm0, xmm0
    mov edx, [ebp + 24]     ; B
    lea eax, [edi * 4]
    add edx, eax            ; &B[0, j]
    mov eax, [esp + 12]     ; row0_A
    xor ebx, ebx
.k_sc_0:
    movss xmm1, [eax + ebx * 4]
    mulss xmm1, [edx]
    addss xmm0, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_sc_0
    mov eax, [esp + 28]
    movss [eax + edi * 4], xmm0

    ; Row 1
    xorps xmm0, xmm0
    mov edx, [ebp + 24]
    lea eax, [edi * 4]
    add edx, eax
    mov eax, [esp + 16]
    xor ebx, ebx
.k_sc_1:
    movss xmm1, [eax + ebx * 4]
    mulss xmm1, [edx]
    addss xmm0, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_sc_1
    mov eax, [esp + 32]
    movss [eax + edi * 4], xmm0

    ; Row 2
    xorps xmm0, xmm0
    mov edx, [ebp + 24]
    lea eax, [edi * 4]
    add edx, eax
    mov eax, [esp + 20]
    xor ebx, ebx
.k_sc_2:
    movss xmm1, [eax + ebx * 4]
    mulss xmm1, [edx]
    addss xmm0, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_sc_2
    mov eax, [esp + 36]
    movss [eax + edi * 4], xmm0

    ; Row 3
    xorps xmm0, xmm0
    mov edx, [ebp + 24]
    lea eax, [edi * 4]
    add edx, eax
    mov eax, [esp + 24]
    xor ebx, ebx
.k_sc_3:
    movss xmm1, [eax + ebx * 4]
    mulss xmm1, [edx]
    addss xmm0, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_sc_3
    mov eax, [esp + 40]
    movss [eax + edi * 4], xmm0

    inc edi
    cmp edi, [ebp + 12]
    jl .col_scalar_loop

.row_4_next:
    add esi, 4
    jmp .row_loop_4

.row_tail_prep:
    cmp esi, [ebp + 8]      ; i >= M?
    jge .matmul_done

.row_tail_loop:
    mov eax, esi
    imul eax, [esp + 8]
    mov edx, [ebp + 20]
    add edx, eax
    mov [esp + 12], edx     ; row_i_A

    mov eax, esi
    imul eax, [esp + 4]
    mov edx, [ebp + 28]
    add edx, eax
    mov [esp + 28], edx     ; row_i_C

    xor edi, edi

.row_tail_col_4:
    lea eax, [edi + 4]
    cmp eax, [ebp + 12]
    jg .row_tail_col_sc

    xorps xmm4, xmm4
    mov edx, [ebp + 24]     ; B
    lea eax, [edi * 4]
    add edx, eax
    mov eax, [esp + 12]     ; row_i_A
    xor ebx, ebx

.k_tail_4:
    movups xmm0, [edx]
    movss xmm1, [eax + ebx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm4, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_tail_4

    mov eax, [esp + 28]
    movups [eax + edi * 4], xmm4
    add edi, 4
    jmp .row_tail_col_4

.row_tail_col_sc:
    cmp edi, [ebp + 12]
    jge .row_tail_next

.row_tail_sc_loop:
    xorps xmm0, xmm0
    mov edx, [ebp + 24]
    lea eax, [edi * 4]
    add edx, eax
    mov eax, [esp + 12]
    xor ebx, ebx

.k_tail_sc:
    movss xmm1, [eax + ebx * 4]
    mulss xmm1, [edx]
    addss xmm0, xmm1
    add edx, [esp + 0]
    inc ebx
    cmp ebx, [ebp + 16]
    jl .k_tail_sc

    mov eax, [esp + 28]
    movss [eax + edi * 4], xmm0
    inc edi
    cmp edi, [ebp + 12]
    jl .row_tail_sc_loop

.row_tail_next:
    inc esi
    cmp esi, [ebp + 8]
    jl .row_tail_loop

.matmul_done:
    add esp, 48
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret

; -----------------------------------------------------------------------------
; void __cdecl nanogemm_sgemm(int M, int N, int K, float alpha,
;                             const float* A, int lda,
;                             const float* B, int ldb,
;                             float beta,
;                             float* C, int ldc)
; Stack layout:
; [ebp + 8]  = M
; [ebp + 12] = N
; [ebp + 16] = K
; [ebp + 20] = alpha
; [ebp + 24] = A
; [ebp + 28] = lda
; [ebp + 32] = B
; [ebp + 36] = ldb
; [ebp + 40] = beta
; [ebp + 44] = C
; [ebp + 48] = ldc
; -----------------------------------------------------------------------------
nanogemm_sgemm:
    push ebp
    mov ebp, esp

    push ebx
    push esi
    push edi

    mov eax, [ebp + 8]      ; M
    test eax, eax
    jle .sgemm_done
    mov eax, [ebp + 12]     ; N
    test eax, eax
    jle .sgemm_done
    mov eax, [ebp + 16]     ; K
    test eax, eax
    jle .sgemm_done

    xor esi, esi            ; i = 0

.sgemm_i_loop:
    ; row_i_A = A + i * lda * 4
    mov eax, [ebp + 28]     ; lda
    imul eax, esi
    shl eax, 2
    mov ecx, [ebp + 24]     ; A
    add ecx, eax            ; row_i_A in ecx

    ; row_i_C = C + i * ldc * 4
    mov eax, [ebp + 48]     ; ldc
    imul eax, esi
    shl eax, 2
    mov edx, [ebp + 44]     ; C
    add edx, eax            ; row_i_C in edx

    xor edi, edi            ; j = 0

.sgemm_j_4:
    lea eax, [edi + 4]
    cmp eax, [ebp + 12]
    jg .sgemm_j_sc

    xorps xmm4, xmm4
    mov eax, [ebp + 32]     ; B
    lea ebx, [edi * 4]
    add eax, ebx            ; &B[0, j]
    mov ebx, [ebp + 36]     ; ldb
    shl ebx, 2              ; ldb * 4

    push esi
    push edx
    xor edx, edx            ; k = 0

.sgemm_k_4:
    movups xmm0, [eax]
    movss xmm1, [ecx + edx * 4]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm4, xmm1
    add eax, ebx
    inc edx
    cmp edx, [ebp + 16]
    jl .sgemm_k_4

    pop edx                 ; restore row_i_C
    pop esi                 ; restore i

    ; Scale by alpha
    movss xmm1, [ebp + 20]
    shufps xmm1, xmm1, 0
    mulps xmm4, xmm1

    ; If beta != 0
    movss xmm0, [ebp + 40]  ; beta
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    je .sgemm_store_4

    shufps xmm0, xmm0, 0
    movups xmm2, [edx + edi * 4]
    mulps xmm2, xmm0
    addps xmm4, xmm2

.sgemm_store_4:
    movups [edx + edi * 4], xmm4
    add edi, 4
    jmp .sgemm_j_4

.sgemm_j_sc:
    cmp edi, [ebp + 12]
    jge .sgemm_i_next

.sgemm_j_sc_loop:
    xorps xmm4, xmm4
    mov eax, [ebp + 32]     ; B
    lea ebx, [edi * 4]
    add eax, ebx
    mov ebx, [ebp + 36]
    shl ebx, 2

    push esi
    push edx
    xor edx, edx

.sgemm_k_sc:
    movss xmm0, [ecx + edx * 4]
    mulss xmm0, [eax]
    addss xmm4, xmm0
    add eax, ebx
    inc edx
    cmp edx, [ebp + 16]
    jl .sgemm_k_sc

    pop edx
    pop esi

    ; Scale by alpha
    mulss xmm4, [ebp + 20]

    ; If beta != 0
    movss xmm0, [ebp + 40]
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    je .sgemm_store_sc

    movss xmm1, [edx + edi * 4]
    mulss xmm1, xmm0
    addss xmm4, xmm1

.sgemm_store_sc:
    movss [edx + edi * 4], xmm4
    inc edi
    cmp edi, [ebp + 12]
    jl .sgemm_j_sc_loop

.sgemm_i_next:
    inc esi
    cmp esi, [ebp + 8]
    jl .sgemm_i_loop

.sgemm_done:
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret
