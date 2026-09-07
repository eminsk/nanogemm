; =============================================================================
; NanoGEMM — High-Performance 32-bit SSE2 Matrix Multiplication DLL (x86 FASM)
; =============================================================================

format PE GUI 4.0 DLL
entry DllEntryPoint

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

section '.text' code readable executable

proc DllEntryPoint hinstDLL, fdwReason, lpvReserved
    mov eax, 1
    ret
endp

; Include matrix multiplication routines
include 'nanogemm32_kernel.inc'

section '.data' data readable
isa_str db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0

section '.edata' export data readable
export 'nanogemm32.dll',\
       nanogemm_matmul,   'nanogemm_matmul',\
       nanogemm_sgemm,    'nanogemm_sgemm',\
       nanogemm_simd_isa, 'nanogemm_simd_isa'
