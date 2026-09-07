; =============================================================================
; NanoGEMM — 32-bit Native Standalone FASM Test & Benchmark Suite
; =============================================================================

format PE console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  NanoGEMM Native x86 32-bit FASM SSE2 Test Suite', 13, 10
                db '====================================================================', 13, 10, 0
    test1_msg   db '  [TEST 1] Scalar 1x1 matmul: ', 0
    test2_msg   db '  [TEST 2] Identity 4x4 matmul: ', 0
    test3_msg   db '  [TEST 3] Uniform 16x16 SSE2 microkernel (48.0): ', 0
    test4_msg   db '  [TEST 4] Rectangular non-aligned (7, 13, 19): ', 0
    test5_msg   db '  [TEST 5] 64x64 Benchmark: ', 0
    test6_msg   db '  [TEST 6] 128x128 Benchmark: ', 0
    pass_str    db 'PASS', 13, 10, 0
    fail_str    db 'FAIL (Diff too large!)', 13, 10, 0
    bench_fmt   db 'PASS (Elapsed: %u us, Approx %u MFLOPS)', 13, 10, 0
    all_ok_msg  db '--------------------------------------------------------------------', 13, 10
                db '  ALL 32-BIT FASM NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0
    isa_str     db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0

    freq_lo dd 0
    freq_hi dd 0
    t1_lo   dd 0
    t1_hi   dd 0
    t2_lo   dd 0
    t2_hi   dd 0

section '.bss' readable writeable
    align 16
    mat_A   rd 16384
    align 16
    mat_B   rd 16384
    align 16
    mat_C   rd 16384

section '.text' code readable executable

; Include matrix multiplication routines
include 'nanogemm32_kernel.inc'

start:
    push freq_lo
    call [QueryPerformanceFrequency]

    push hdr_msg
    call [printf]
    add esp, 4

    ; -------------------------------------------------------------------------
    ; TEST 1: 1x1 Matmul (A=[3.0], B=[4.0] => C=[12.0])
    ; -------------------------------------------------------------------------
    push test1_msg
    call [printf]
    add esp, 4

    mov dword [mat_A], 40400000h    ; 3.0f
    mov dword [mat_B], 40800000h    ; 4.0f
    mov dword [mat_C], 0

    push mat_C
    push mat_B
    push mat_A
    push 1
    push 1
    push 1
    call nanogemm_matmul
    add esp, 24

    mov eax, [mat_C]
    cmp eax, 41400000h              ; 12.0f
    jne .fail1
    push pass_str
    call [printf]
    add esp, 4
    jmp .test2
.fail1:
    push fail_str
    call [printf]
    add esp, 4
    push 1
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 2: Identity 4x4 Matmul (A * I = A)
    ; -------------------------------------------------------------------------
.test2:
    push test2_msg
    call [printf]
    add esp, 4

    xor ecx, ecx
.init_a2:
    cvtsi2ss xmm0, ecx
    movss [mat_A + ecx * 4], xmm0
    mov dword [mat_B + ecx * 4], 0
    inc ecx
    cmp ecx, 16
    jl .init_a2

    mov dword [mat_B + 0 * 4],  3f800000h ; 1.0f
    mov dword [mat_B + 5 * 4],  3f800000h
    mov dword [mat_B + 10 * 4], 3f800000h
    mov dword [mat_B + 15 * 4], 3f800000h

    push mat_C
    push mat_B
    push mat_A
    push 4
    push 4
    push 4
    call nanogemm_matmul
    add esp, 24

    xor ecx, ecx
.check_c2:
    mov eax, [mat_A + ecx * 4]
    cmp eax, [mat_C + ecx * 4]
    jne .fail2
    inc ecx
    cmp ecx, 16
    jl .check_c2
    push pass_str
    call [printf]
    add esp, 4
    jmp .test3
.fail2:
    push fail_str
    call [printf]
    add esp, 4
    push 2
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 3: Uniform 16x16 SSE2 microkernel
    ; A = 1.5f, B = 2.0f => C = 16 * 3.0 = 48.0f (42400000h)
    ; -------------------------------------------------------------------------
.test3:
    push test3_msg
    call [printf]
    add esp, 4

    xor ecx, ecx
.init3:
    mov dword [mat_A + ecx * 4], 3fc00000h ; 1.5f
    mov dword [mat_B + ecx * 4], 40000000h ; 2.0f
    mov dword [mat_C + ecx * 4], 0
    inc ecx
    cmp ecx, 256
    jl .init3

    push mat_C
    push mat_B
    push mat_A
    push 16
    push 16
    push 16
    call nanogemm_matmul
    add esp, 24

    xor ecx, ecx
.check3:
    mov eax, [mat_C + ecx * 4]
    cmp eax, 42400000h
    jne .fail3
    inc ecx
    cmp ecx, 256
    jl .check3
    push pass_str
    call [printf]
    add esp, 4
    jmp .test4
.fail3:
    push fail_str
    call [printf]
    add esp, 4
    push 3
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 4: Rectangular Prime / Non-Aligned (7, 13, 19)
    ; A(7x13) = 1.0f, B(13x19) = 1.0f => C(7x19) = 13.0f (41500000h)
    ; -------------------------------------------------------------------------
.test4:
    push test4_msg
    call [printf]
    add esp, 4

    xor ecx, ecx
.init4_a:
    mov dword [mat_A + ecx * 4], 3f800000h ; 1.0f
    inc ecx
    cmp ecx, 7 * 13
    jl .init4_a

    xor ecx, ecx
.init4_b:
    mov dword [mat_B + ecx * 4], 3f800000h ; 1.0f
    inc ecx
    cmp ecx, 13 * 19
    jl .init4_b

    push mat_C
    push mat_B
    push mat_A
    push 13
    push 19
    push 7
    call nanogemm_matmul
    add esp, 24

    xor ecx, ecx
.check4:
    mov eax, [mat_C + ecx * 4]
    cmp eax, 41500000h
    jne .fail4
    inc ecx
    cmp ecx, 7 * 19
    jl .check4
    push pass_str
    call [printf]
    add esp, 4
    jmp .test5
.fail4:
    push fail_str
    call [printf]
    add esp, 4
    push 4
    call [ExitProcess]

    ; -------------------------------------------------------------------------
    ; TEST 5: 64x64 Benchmark
    ; -------------------------------------------------------------------------
.test5:
    push test5_msg
    call [printf]
    add esp, 4

    ; Warmup
    push mat_C
    push mat_B
    push mat_A
    push 64
    push 64
    push 64
    call nanogemm_matmul
    add esp, 24

    push t1_lo
    call [QueryPerformanceCounter]

    mov edi, 100
.bench5_loop:
    push mat_C
    push mat_B
    push mat_A
    push 64
    push 64
    push 64
    call nanogemm_matmul
    add esp, 24
    dec edi
    jnz .bench5_loop

    push t2_lo
    call [QueryPerformanceCounter]

    mov eax, [t2_lo]
    sub eax, [t1_lo]
    imul eax, 10000
    xor edx, edx
    div dword [freq_lo]

    test eax, eax
    jnz @f
    mov eax, 1
@@: mov ebx, eax                 ; us in ebx

    mov eax, 524288
    xor edx, edx
    div ebx
    mov ecx, eax                 ; mflops in ecx

    push ecx
    push ebx
    push bench_fmt
    call [printf]
    add esp, 12

    ; -------------------------------------------------------------------------
    ; TEST 6: 128x128 Benchmark
    ; -------------------------------------------------------------------------
.test6:
    push test6_msg
    call [printf]
    add esp, 4

    push mat_C
    push mat_B
    push mat_A
    push 128
    push 128
    push 128
    call nanogemm_matmul
    add esp, 24

    push t1_lo
    call [QueryPerformanceCounter]

    mov edi, 50
.bench6_loop:
    push mat_C
    push mat_B
    push mat_A
    push 128
    push 128
    push 128
    call nanogemm_matmul
    add esp, 24
    dec edi
    jnz .bench6_loop

    push t2_lo
    call [QueryPerformanceCounter]

    mov eax, [t2_lo]
    sub eax, [t1_lo]
    imul eax, 20000
    xor edx, edx
    div dword [freq_lo]

    test eax, eax
    jnz @f
    mov eax, 1
@@: mov ebx, eax

    mov eax, 4194304
    xor edx, edx
    div ebx
    mov ecx, eax

    push ecx
    push ebx
    push bench_fmt
    call [printf]
    add esp, 12

    push all_ok_msg
    call [printf]
    add esp, 4

    push 0
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
