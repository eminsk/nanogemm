; =============================================================================
; NanoGEMM — High-Performance AVX2+FMA Matrix Multiplication DLL (x86-64 FASM)
; =============================================================================

format PE64 GUI 6.0 DLL
entry DllEntryPoint

include 'C:\proekts\FASM\INCLUDE\WIN64A.INC'

section '.text' code readable executable

proc DllEntryPoint hinstDLL, fdwReason, lpvReserved
    mov eax, 1
    ret
endp

; -----------------------------------------------------------------------------
; const char* nanogemm_simd_isa(void)
; -----------------------------------------------------------------------------
nanogemm_simd_isa:
    lea rax, [isa_str]
    ret

; Include matrix multiplication routines
include 'nanogemm64_kernel.inc'

section '.data' data readable
isa_str db 'AVX2+FMA (FASM x86-64, 256-bit SIMD)', 0

section '.edata' export data readable
export 'nanogemm64.dll',\
       nanogemm_matmul,   'nanogemm_matmul',\
       nanogemm_sgemm,    'nanogemm_sgemm',\
       nanogemm_simd_isa, 'nanogemm_simd_isa'
