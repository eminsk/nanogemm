"""
Comprehensive Tests for NanoGEMM FASM 32-bit and 64-bit Microkernels.
Tests both native standalone executables and x86-64 DLL via ctypes against NumPy.
"""

import os
import sys
import ctypes
import subprocess
import time
from pathlib import Path
import numpy as np

ASM_DIR = Path(__file__).resolve().parent.parent / "asm"
DLL64_PATH = ASM_DIR / "nanogemm64.dll"
EXE64_PATH = ASM_DIR / "test_nanogemm64.exe"
DLL32_PATH = ASM_DIR / "nanogemm32.dll"
EXE32_PATH = ASM_DIR / "test_nanogemm32.exe"
BUILD_BAT = ASM_DIR / "build.bat"


def ensure_binaries_built():
    """Builds FASM binaries using build.bat if any target is missing."""
    needed = [DLL64_PATH, EXE64_PATH, DLL32_PATH, EXE32_PATH]
    if any(not p.exists() for p in needed):
        print("  [Build] Binaries missing. Compiling with build.bat...")
        res = subprocess.run(["cmd.exe", "/c", str(BUILD_BAT)], cwd=str(ASM_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Assembly build failed:\n{res.stdout}\n{res.stderr}"
        print("  [Build] Build completed successfully.")


def test_fasm64_standalone_exe():
    """Runs the 64-bit native PE console test suite."""
    ensure_binaries_built()
    assert EXE64_PATH.exists(), f"Executable not found: {EXE64_PATH}"
    res = subprocess.run([str(EXE64_PATH)], capture_output=True, text=True)
    print(f"\n--- Output of {EXE64_PATH.name} ---")
    print(res.stdout)
    assert res.returncode == 0, f"64-bit standalone test failed with returncode {res.returncode}"
    assert "ALL 64-BIT FASM NATIVE TESTS PASSED" in res.stdout
    print("  [PASS] test_nanogemm64.exe completed with 100% accuracy.")


def test_fasm32_standalone_exe():
    """Runs the 32-bit native PE console test suite under WoW64."""
    ensure_binaries_built()
    assert EXE32_PATH.exists(), f"Executable not found: {EXE32_PATH}"
    res = subprocess.run([str(EXE32_PATH)], capture_output=True, text=True)
    print(f"\n--- Output of {EXE32_PATH.name} ---")
    print(res.stdout)
    assert res.returncode == 0, f"32-bit standalone test failed with returncode {res.returncode}"
    assert "ALL 32-BIT FASM NATIVE TESTS PASSED" in res.stdout
    print("  [PASS] test_nanogemm32.exe completed with 100% accuracy.")


def test_fasm64_dll_accuracy_vs_numpy():
    """Loads nanogemm64.dll and verifies correctness vs NumPy for square and rectangular matrices."""
    ensure_binaries_built()
    dll = ctypes.CDLL(str(DLL64_PATH))

    dll.nanogemm_simd_isa.restype = ctypes.c_char_p
    isa = dll.nanogemm_simd_isa().decode("utf-8")
    print(f"\n  [Loaded FASM 64-bit DLL ISA]: {isa}")
    assert "AVX2" in isa

    dll.nanogemm_matmul.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    dll.nanogemm_matmul.restype = None

    # 1. Square dimensions
    square_dims = [1, 2, 3, 4, 7, 8, 15, 16, 24, 32, 48, 64, 128, 256]
    for dim in square_dims:
        np.random.seed(42 + dim)
        A = np.random.randn(dim, dim).astype(np.float32)
        B = np.random.randn(dim, dim).astype(np.float32)
        C = np.zeros((dim, dim), dtype=np.float32)

        C_expected = A @ B
        dll.nanogemm_matmul(dim, dim, dim, A.ctypes.data, B.ctypes.data, C.ctypes.data)

        max_err = float(np.max(np.abs(C - C_expected)))
        assert np.allclose(C, C_expected, atol=1e-4, rtol=1e-4), (
            f"Mismatch on {dim}x{dim}: max_err={max_err}"
        )
        print(f"  [PASS] Square {dim:3d}x{dim:3d} (max diff: {max_err:.2e})")

    # 2. Rectangular and Prime dimensions
    rect_shapes = [
        (1, 128, 1),
        (64, 1, 64),
        (32, 64, 128),
        (128, 32, 64),
        (7, 13, 19),
        (37, 59, 41),
        (65, 129, 65),
        (127, 255, 127)
    ]
    for M, K, N in rect_shapes:
        np.random.seed(1337 + M + N + K)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        C_expected = A @ B
        dll.nanogemm_matmul(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data)

        max_err = float(np.max(np.abs(C - C_expected)))
        assert np.allclose(C, C_expected, atol=1e-4, rtol=1e-4), (
            f"Mismatch on ({M},{K}) @ ({K},{N}): max_err={max_err}"
        )
        print(f"  [PASS] Rectangular ({M:3d},{K:3d}) @ ({K:3d},{N:3d}) (max diff: {max_err:.2e})")


def test_fasm64_dll_sgemm_alpha_beta():
    """Verifies BLAS SGEMM with alpha, beta, and leading dimensions."""
    ensure_binaries_built()
    dll = ctypes.CDLL(str(DLL64_PATH))

    dll.nanogemm_sgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int
    ]
    dll.nanogemm_sgemm.restype = None

    test_cases = [
        (16, 16, 16, 2.5, 1.25),
        (32, 32, 32, 1.0, 0.0),
        (64, 64, 64, 3.14, 0.5),
        (19, 23, 29, 0.75, 2.0),
        (1, 64, 1, 1.5, 1.0),
        (37, 19, 41, 2.0, 1.0)
    ]
    for M, K, N, alpha, beta in test_cases:
        np.random.seed(999 + M + N)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.random.randn(M, N).astype(np.float32)

        C_expected = alpha * (A @ B) + beta * C
        C_actual = C.copy()

        dll.nanogemm_sgemm(
            M, N, K,
            alpha,
            A.ctypes.data, K,
            B.ctypes.data, N,
            beta,
            C_actual.ctypes.data, N
        )

        max_err = float(np.max(np.abs(C_actual - C_expected)))
        assert np.allclose(C_actual, C_expected, atol=1e-4, rtol=1e-4), (
            f"Mismatch on sgemm ({M},{K},{N}): max_err={max_err}"
        )
        print(f"  [PASS] SGEMM ({M:2d},{K:2d},{N:2d}) alpha={alpha} beta={beta} (max diff: {max_err:.2e})")


def run_benchmark():
    """Runs a quick performance benchmark comparing FASM x64 vs NumPy."""
    ensure_binaries_built()
    dll = ctypes.CDLL(str(DLL64_PATH))
    dll.nanogemm_matmul.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    print("\n" + "=" * 80)
    print("  NanoGEMM FASM x64 AVX2 Microkernel Benchmark vs NumPy CPU")
    print("=" * 80)
    print(f"{'Matrix Size':<12} | {'NumPy (us)':<12} | {'FASM x64 (us)':<14} | {'Speedup':<14} | {'FASM GFLOPS'}")
    print("-" * 80)

    for dim in [16, 32, 64, 128, 256]:
        A = np.random.randn(dim, dim).astype(np.float32)
        B = np.random.randn(dim, dim).astype(np.float32)
        C = np.zeros((dim, dim), dtype=np.float32)

        # Warmup
        for _ in range(5):
            _ = A @ B
            dll.nanogemm_matmul(dim, dim, dim, A.ctypes.data, B.ctypes.data, C.ctypes.data)

        # NumPy timing
        iters = 500 if dim <= 64 else 100
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = A @ B
        numpy_us = (time.perf_counter() - t0) * 1e6 / iters

        # FASM timing
        t0 = time.perf_counter()
        for _ in range(iters):
            dll.nanogemm_matmul(dim, dim, dim, A.ctypes.data, B.ctypes.data, C.ctypes.data)
        fasm_us = (time.perf_counter() - t0) * 1e6 / iters

        speedup = numpy_us / fasm_us
        speedup_str = f"{speedup:.2f}x {'(FASTER)' if speedup > 1.0 else ''}"
        gflops = (2.0 * (dim ** 3)) / (fasm_us * 1e-6) / 1e9

        print(f"{f'{dim}x{dim}':<12} | {numpy_us:>9.2f} us | {fasm_us:>11.2f} us | {speedup_str:<14} | {gflops:>9.2f} GFLOPS")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Running NanoGEMM FASM 32-bit and 64-bit Test Suite")
    print("=" * 65)

    print("\n>>> 1. Testing 64-bit Standalone Native Executable:")
    test_fasm64_standalone_exe()

    print("\n>>> 2. Testing 32-bit Standalone Native Executable (WoW64):")
    test_fasm32_standalone_exe()

    print("\n>>> 3. Testing 64-bit DLL (ctypes) vs NumPy Reference:")
    test_fasm64_dll_accuracy_vs_numpy()

    print("\n>>> 4. Testing 64-bit BLAS SGEMM (Alpha & Beta Scaling):")
    test_fasm64_dll_sgemm_alpha_beta()

    print("\n>>> 5. Benchmark Performance:")
    run_benchmark()

    print("ALL FASM TESTS PASSED WITH 100% NUMERICAL ACCURACY!")
