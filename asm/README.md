# NanoGEMM — Native Flat Assembler (FASM) Acceleration Suite ?

Direct hardware assembly implementation of single-precision matrix multiplication (SGEMM) for both **x86-64 (64-bit AVX2+FMA)** and **x86 (32-bit SSE2)** architectures, written with [Flat Assembler (FASM)](https://flatassembler.net/).

---

## ?? Architecture & Components

```
asm/
+-- nanogemm64_kernel.inc   # 64-bit AVX2+FMA register-tiled (4x16, 4x8) microkernels & SGEMM driver
+-- nanogemm64.asm          # PE64 DLL source exporting nanogemm_matmul, nanogemm_sgemm
+-- nanogemm64.dll          # Compiled 64-bit Windows DLL (callable from C, C++, Python ctypes)
+-- test_nanogemm64.asm     # Standalone PE64 console benchmark & self-test suite
+-- test_nanogemm64.exe     # Compiled 64-bit native executable (zero runtime dependencies)
+-- nanogemm32_kernel.inc   # 32-bit SSE2 register-tiled (4x4) microkernels & SGEMM driver
+-- nanogemm32.asm          # PE32 DLL source exporting nanogemm_matmul, nanogemm_sgemm
+-- nanogemm32.dll          # Compiled 32-bit Windows DLL (cdecl calling convention)
+-- test_nanogemm32.asm     # Standalone PE32 console benchmark & self-test suite (runs via WoW64)
+-- test_nanogemm32.exe     # Compiled 32-bit native executable (zero runtime dependencies)
L-- build.bat               # Automated build & verification script
```

---

## ?? Key Features

### 1. 64-bit Engine (`nanogemm64.dll`, `test_nanogemm64.exe`)
- **ISA Target:** x86-64 with AVX2 (256-bit SIMD) and FMA3 (`vfmadd231ps`, `vbroadcastss`, `vmovups`).
- **Microkernel:** $4 \times 16$ and $4 \times 8$ register-tiled loop using 8 YMM accumulators (`ymm4`–`ymm11`).
- **Edge Kernel:** Handles arbitrary, odd, or prime matrix dimensions ($M, N, K$) with zero memory padding or buffer copying.
- **Calling Convention:** Complies strictly with the **Microsoft x64 ABI**, preserving all non-volatile registers (`RBX`, `RSI`, `RDI`, `R12`–`R15`, `XMM6`–`XMM15`).
- **Throughput:** Exceeds **38–40 GFLOPS** on a single CPU core.

### 2. 32-bit Engine (`nanogemm32.dll`, `test_nanogemm32.exe`)
- **ISA Target:** x86 32-bit with SSE2 (128-bit SIMD, `movups`, `shufps`, `mulps`, `addps`).
- **Microkernel:** $4 \times 4$ register-tiled loop using 4 XMM accumulators (`xmm4`–`xmm7`).
- **Compatibility:** Runs on any 32-bit x86 OS or 64-bit Windows via WoW64 with zero external dependencies.
- **Calling Convention:** Standard `cdecl` calling convention.
- **Throughput:** Delivers **13–15 GFLOPS** in pure 32-bit mode.

---

## ?? Building from Source

To assemble all targets and run the native verification suites, run:

```cmd
cd asm
build.bat
```

Or assemble manually using `FASM.EXE`:

```cmd
:: 64-bit targets
fasm nanogemm64.asm nanogemm64.dll
fasm test_nanogemm64.asm test_nanogemm64.exe

:: 32-bit targets
fasm nanogemm32.asm nanogemm32.dll
fasm test_nanogemm32.asm test_nanogemm32.exe
```

---

## ?? Running Tests

### 1. Native Executable Tests
```cmd
:: Run 64-bit native test suite
test_nanogemm64.exe

:: Run 32-bit native test suite (WoW64)
test_nanogemm32.exe
```

### 2. Python Automated Verification vs NumPy
```bash
python tests/test_fasm.py
```
Tests numerical accuracy across:
- Square shapes from $1 \times 1$ to $256 \times 256$
- Rectangular and prime shapes: $(7, 13, 19)$, $(37, 59, 41)$, $(65, 129, 65)$, $(127, 255, 127)$
- Full BLAS SGEMM with $\alpha$ and $\beta$ scaling
- Reports relative and absolute diffs (`np.allclose(atol=1e-4)`) and GFLOPS.
