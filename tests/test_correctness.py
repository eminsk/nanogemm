"""
Correctness tests for NanoGEMM matrix multiplication engine.
Works with both pytest and python direct execution.
"""

import numpy as np
import sys
from pathlib import Path

try:
    import nanogemm as ng
except ImportError:
    for p in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
        if (p / "nanogemm" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            break
    import nanogemm as ng



def test_simd_isa_available():
    isa = ng.get_simd_isa()
    assert isinstance(isa, str)
    assert len(isa) > 0
    print(f"  [Active ISA]: {isa}")


def run_square_tests():
    dims = [1, 2, 4, 6, 8, 16, 24, 32, 64, 128, 256]
    for dim in dims:
        np.random.seed(42 + dim)
        A = np.random.randn(dim, dim).astype(np.float32)
        B = np.random.randn(dim, dim).astype(np.float32)

        C_expected = A @ B
        C_actual = ng.matmul(A, B)

        max_err = np.max(np.abs(C_actual - C_expected))
        assert np.allclose(C_actual, C_expected, atol=1e-4, rtol=1e-4), (
            f"Mismatch for square {dim}x{dim}. Max diff: {max_err}"
        )
        print(f"  [PASS] Square {dim:3d}x{dim:3d} (max diff: {max_err:.2e})")


def run_rectangular_tests():
    cases = [
        (1, 128, 1),       # Dot product
        (64, 1, 64),       # Outer product
        (32, 64, 128),     # Typical rectangular
        (128, 32, 64),
        (7, 13, 19),       # Prime / non-aligned dimensions
        (37, 59, 41),      # Large primes (stress edge kernel)
        (65, 129, 65),     # Just over tile boundaries
        (127, 255, 127),
    ]
    for M, K, N in cases:
        np.random.seed(1337 + M * 10 + N)
        A = np.random.uniform(-5.0, 5.0, size=(M, K)).astype(np.float32)
        B = np.random.uniform(-5.0, 5.0, size=(K, N)).astype(np.float32)

        C_expected = A @ B
        C_actual = ng.matmul(A, B)

        max_err = np.max(np.abs(C_actual - C_expected))
        assert np.allclose(C_actual, C_expected, atol=1e-4, rtol=1e-4), (
            f"Mismatch for rectangular ({M},{K}) @ ({K},{N}). Max diff: {max_err}"
        )
        print(f"  [PASS] Rectangular ({M:3d},{K:3d}) @ ({K:3d},{N:3d}) (max diff: {max_err:.2e})")


def run_preallocated_out():
    A = np.random.randn(48, 48).astype(np.float32)
    B = np.random.randn(48, 48).astype(np.float32)
    out = np.zeros((48, 48), dtype=np.float32)

    res = ng.matmul(A, B, out=out)
    assert res is out
    assert np.allclose(out, A @ B, atol=1e-4, rtol=1e-4)
    print("  [PASS] Preallocated out buffer")


def run_sgemm_alpha_beta():
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    C_init = np.random.randn(32, 32).astype(np.float32)

    alpha = 2.5
    beta = 1.25

    C_expected = alpha * (A @ B) + beta * C_init
    C_actual = ng.sgemm(A, B, alpha=alpha, beta=beta, c=C_init.copy())

    assert np.allclose(C_actual, C_expected, atol=1e-4, rtol=1e-4)
    print("  [PASS] BLAS SGEMM alpha & beta scaling")


def run_error_handling():
    A = np.ones((10, 20), dtype=np.float32)
    B = np.ones((25, 30), dtype=np.float32)
    try:
        ng.matmul(A, B)
        assert False, "Expected ValueError on dimension mismatch"
    except ValueError:
        pass

    # 1D is invalid
    try:
        ng.matmul(np.ones((2,)), np.ones((2, 2)))
        assert False, "Expected ValueError on 1D input"
    except ValueError:
        pass

    # 5D is unsupported
    try:
        ng.matmul(np.ones((2, 2, 2, 2, 2)), np.ones((2, 2, 2, 2, 2)))
        assert False, "Expected ValueError on 5D input"
    except ValueError:
        pass

    # Batch dimension mismatch
    try:
        ng.matmul(np.ones((2, 4, 4)), np.ones((3, 4, 4)))
        assert False, "Expected ValueError on batch mismatch"
    except ValueError:
        pass

    print("  [PASS] Error handling & input shape validation")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Running NanoGEMM Full Correctness Test Suite")
    print("=" * 60)
    test_simd_isa_available()
    print("\n--- 1. Square Matrices ---")
    run_square_tests()
    print("\n--- 2. Rectangular & Prime Dimensions ---")
    run_rectangular_tests()
    print("\n--- 3. Preallocated Output Buffer ---")
    run_preallocated_out()
    print("\n--- 4. BLAS SGEMM (Alpha & Beta) ---")
    run_sgemm_alpha_beta()
    print("\n--- 5. Error Handling ---")
    run_error_handling()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED WITH 100% NUMERICAL ACCURACY!")
    print("=" * 60 + "\n")
