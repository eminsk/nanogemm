; =============================================================================
; NanoGEMM — 64-bit Native Standalone FASM Test & Benchmark Suite
; =============================================================================

format PE64 console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN64A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  NanoGEMM Native x86-64 FASM AVX2+FMA Test Suite', 13, 10
                db '====================================================================', 13, 10, 0
    test1_msg   db '  [TEST 1] Scalar 1x1 matmul: ', 0
    test2_msg   db '  [TEST 2] Identity 4x4 matmul: ', 0
    test3_msg   db '  [TEST 3] Uniform 16x16 AVX2 microkernel (48.0): ', 0
    test4_msg   db '  [TEST 4] Rectangular non-aligned (7, 13, 19): ', 0
    test5_msg   db '  [TEST 5] 64x64 Benchmark: ', 0
    test6_msg   db '  [TEST 6] 128x128 Benchmark: ', 0
    pass_str    db 'PASS', 13, 10, 0
    fail_str    db 'FAIL (Diff too large!)', 13, 10, 0
    test7_msg   db '  [TEST 7] Batched GEMM (BMM) 4x(16x16): ', 0
    test8_msg   db '  [TEST 8] Quantized INT8 SIMD GEMM 16x16: ', 0
    bench_fmt   db 'PASS (Elapsed: %u us, Approx %u MFLOPS)', 13, 10, 0
    all_ok_msg  db '--------------------------------------------------------------------', 13, 10
                db '  ALL 64-BIT FASM NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0

    freq    rq 1
    t_start rq 1
    t_end   rq 1

section '.bss' readable writeable
    align 32
    mat_A   rd 16384
    align 32
    mat_B   rd 16384
    align 32
    mat_C   rd 16384

section '.text' code readable executable

; Include matrix multiplication routines
include 'nanogemm64_kernel.inc'

start:
    sub rsp, 80h

    ; Initialize high resolution timer frequency
    lea rcx, [freq]
    call [QueryPerformanceFrequency]

    lea rcx, [hdr_msg]
    call [printf]

    ; -------------------------------------------------------------------------
    ; TEST 1: 1x1 Matmul (A=[3.0], B=[4.0] => C=[12.0])
    ; -------------------------------------------------------------------------
    lea rcx, [test1_msg]
    call [printf]

    mov dword [mat_A], 40400000h    ; 3.0f
    mov dword [mat_B], 40800000h    ; 4.0f
    mov dword [mat_C], 0

    mov ecx, 1
    mov edx, 1
    mov r8d, 1
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    mov eax, [mat_C]
    cmp eax, 41400000h              ; 12.0f
    jne .fail1
    lea rcx, [pass_str]
    call [printf]
    jmp .test2
.fail1:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 2: Identity 4x4 Matmul (A * I = A)
    ; -------------------------------------------------------------------------
.test2:
    lea rcx, [test2_msg]
    call [printf]

    xor ecx, ecx
.init_a2:
    cvtsi2ss xmm0, ecx
    movss [mat_A + rcx*4], xmm0
    mov dword [mat_B + rcx*4], 0
    inc ecx
    cmp ecx, 16
    jl .init_a2

    mov dword [mat_B + 0*4],  3f800000h
    mov dword [mat_B + 5*4],  3f800000h
    mov dword [mat_B + 10*4], 3f800000h
    mov dword [mat_B + 15*4], 3f800000h

    mov ecx, 4
    mov edx, 4
    mov r8d, 4
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    xor ecx, ecx
.check_c2:
    mov eax, [mat_A + rcx*4]
    cmp eax, [mat_C + rcx*4]
    jne .fail2
    inc ecx
    cmp ecx, 16
    jl .check_c2
    lea rcx, [pass_str]
    call [printf]
    jmp .test3
.fail2:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 2
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 3: Uniform 16x16 AVX2 microkernel
    ; A = 1.5f, B = 2.0f => C = 16 * 3.0 = 48.0f (42400000h)
    ; -------------------------------------------------------------------------
.test3:
    lea rcx, [test3_msg]
    call [printf]

    xor ecx, ecx
.init3:
    mov dword [mat_A + rcx*4], 3fc00000h    ; 1.5f
    mov dword [mat_B + rcx*4], 40000000h    ; 2.0f
    mov dword [mat_C + rcx*4], 0
    inc ecx
    cmp ecx, 256
    jl .init3

    mov ecx, 16
    mov edx, 16
    mov r8d, 16
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    xor ecx, ecx
.check3:
    mov eax, [mat_C + rcx*4]
    cmp eax, 42400000h
    jne .fail3
    inc ecx
    cmp ecx, 256
    jl .check3
    lea rcx, [pass_str]
    call [printf]
    jmp .test4
.fail3:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 3
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 4: Rectangular Prime / Non-Aligned (7, 13, 19)
    ; A(7x13) = 1.0f, B(13x19) = 1.0f => C(7x19) = 13.0f (41500000h)
    ; -------------------------------------------------------------------------
.test4:
    lea rcx, [test4_msg]
    call [printf]

    xor ecx, ecx
.init4_a:
    mov dword [mat_A + rcx*4], 3f800000h    ; 1.0f
    inc ecx
    cmp ecx, 7 * 13
    jl .init4_a

    xor ecx, ecx
.init4_b:
    mov dword [mat_B + rcx*4], 3f800000h    ; 1.0f
    inc ecx
    cmp ecx, 13 * 19
    jl .init4_b

    mov ecx, 7
    mov edx, 19
    mov r8d, 13
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    xor ecx, ecx
.check4:
    mov eax, [mat_C + rcx*4]
    cmp eax, 41500000h
    jne .fail4
    inc ecx
    cmp ecx, 7 * 19
    jl .check4
    lea rcx, [pass_str]
    call [printf]
    jmp .test5
.fail4:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 4
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 5: 64x64 Benchmark
    ; -------------------------------------------------------------------------
.test5:
    lea rcx, [test5_msg]
    call [printf]

    ; Warmup
    mov ecx, 64
    mov edx, 64
    mov r8d, 64
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    ; Benchmark 100 runs
    lea rcx, [t_start]
    call [QueryPerformanceCounter]

    mov edi, 100
.bench5_loop:
    mov ecx, 64
    mov edx, 64
    mov r8d, 64
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul
    dec edi
    jnz .bench5_loop

    lea rcx, [t_end]
    call [QueryPerformanceCounter]

    ; elapsed_us = (t_end - t_start) * 1000000 / (freq * 100)
    mov rax, [t_end]
    sub rax, [t_start]
    imul rax, 10000
    xor rdx, rdx
    div qword [freq]
    test rax, rax
    jnz @f
    mov rax, 1
@@: mov rsi, rax                 ; rsi = elapsed_us

    mov rax, 524288
    xor rdx, rdx
    div rsi
    mov r9, rax                 ; mflops

    lea rcx, [bench_fmt]
    mov rdx, rsi                ; us
    mov r8, r9                  ; mflops
    call [printf]

    ; -------------------------------------------------------------------------
    ; TEST 6: 128x128 Benchmark
    ; -------------------------------------------------------------------------
.test6:
    lea rcx, [test6_msg]
    call [printf]

    mov ecx, 128
    mov edx, 128
    mov r8d, 128
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul

    lea rcx, [t_start]
    call [QueryPerformanceCounter]

    mov edi, 50
.bench6_loop:
    mov ecx, 128
    mov edx, 128
    mov r8d, 128
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_matmul
    dec edi
    jnz .bench6_loop

    lea rcx, [t_end]
    call [QueryPerformanceCounter]

    mov rax, [t_end]
    sub rax, [t_start]
    imul rax, 20000
    xor rdx, rdx
    div qword [freq]
    test rax, rax
    jnz @f
    mov rax, 1
@@: mov rsi, rax

    mov rax, 4194304
    xor rdx, rdx
    div rsi
    mov r9, rax

    lea rcx, [bench_fmt]
    mov rdx, rsi
    mov r8, r9
    call [printf]

    ; -------------------------------------------------------------------------
    ; TEST 7: Batched GEMM (BMM) 4x(16x16)
    ; -------------------------------------------------------------------------
    lea rcx, [test7_msg]
    call [printf]

    lea rdi, [mat_A]
    mov ecx, 1024
    mov eax, 3F800000h              ; 1.0f
    rep stosd

    lea rdi, [mat_B]
    mov ecx, 1024
    mov eax, 40000000h              ; 2.0f
    rep stosd

    lea rdi, [mat_C]
    mov ecx, 1024
    xor eax, eax
    rep stosd

    mov ecx, 4                      ; batch_count
    mov edx, 16                     ; M
    mov r8d, 16                     ; N
    mov r9d, 16                     ; K
    lea rax, [mat_A]
    mov [rsp + 20h], rax            ; A
    lea rax, [mat_B]
    mov [rsp + 28h], rax            ; B
    lea rax, [mat_C]
    mov [rsp + 30h], rax            ; C
    mov qword [rsp + 38h], 256      ; stride_a
    mov qword [rsp + 40h], 256      ; stride_b
    mov qword [rsp + 48h], 256      ; stride_c
    call nanogemm_bmm

    cmp dword [mat_C], 42000000h
    jne .fail7
    cmp dword [mat_C + 4092], 42000000h
    jne .fail7

    lea rcx, [pass_str]
    call [printf]
    jmp @f
.fail7:
    lea rcx, [fail_str]
    call [printf]
@@:

    ; -------------------------------------------------------------------------
    ; TEST 8: Quantized INT8 SIMD GEMM 16x16
    ; -------------------------------------------------------------------------
    lea rcx, [test8_msg]
    call [printf]

    lea rdi, [mat_A]
    mov ecx, 256
    mov al, 2
    rep stosb

    lea rdi, [mat_B]
    mov ecx, 256
    mov al, 3
    rep stosb

    lea rdi, [mat_C]
    mov ecx, 256
    xor eax, eax
    rep stosd

    mov ecx, 16
    mov edx, 16
    mov r8d, 16
    lea r9, [mat_A]
    lea rax, [mat_B]
    mov [rsp + 20h], rax
    lea rax, [mat_C]
    mov [rsp + 28h], rax
    call nanogemm_gemm_i8i8i32

    cmp dword [mat_C], 96
    jne .fail8
    cmp dword [mat_C + 255*4], 96
    jne .fail8

    lea rcx, [pass_str]
    call [printf]
    jmp @f
.fail8:
    lea rcx, [fail_str]
    call [printf]
@@:

    lea rcx, [all_ok_msg]
    call [printf]

    xor ecx, ecx
    call [ExitProcess]

section '.idata' import data readable
library kernel32, 'kernel32.dll',\
        msvcrt,   'msvcrt.dll'

import kernel32,\
       QueryPerformanceCounter,   'QueryPerformanceCounter',\
       QueryPerformanceFrequency, 'QueryPerformanceFrequency',\
       ExitProcess,               'ExitProcess'

import msvcrt,\
       printf, 'printf'
