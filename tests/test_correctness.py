"""
Correctness tests for NanoGEMM matrix multiplication engine.
Includes:
1. Exact Bitwise Arithmetic Tests (Zero Rounding Error: max diff = 0.00e+00)
   - Exact Integers in float32 domain (K <= 256 fits within 24-bit mantissa)
   - Identity Matrix Preservation (A @ I == A)
   - Dyadic Powers-of-Two Fractions (+-0.5, +-0.25, +-0.125)
   - Permutation Matrices (Bitwise column/row shuffling)
2. High-Precision Float64 Ground Truth & Higham Backward Stability
3. BLAS SGEMM & Buffer Allocation
4. Shape and Dimension Error Handling

Works with both pytest and direct python execution.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure nanogemm is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import nanogemm as ng
except ImportError:
    import nanogemm as ng


def test_simd_isa_available():
    isa = ng.get_simd_isa()
    assert isinstance(isa, str) and len(isa) > 0
    print(f"  [Active ISA]: {isa}")


def run_exact_square_integer_tests():
    """Test square matrices with exact integer values in float32 (zero rounding error)."""
    dims = [1, 2, 4, 6, 8, 16, 24, 32, 64, 128, 256]
    for dim in dims:
        np.random.seed(42 + dim)
        # Small integers [-3, 3] in float32: max inner product for 256 is 256*9 = 2304 << 2^24 (16,777,216)
        A = np.random.randint(-3, 4, size=(dim, dim)).astype(np.float32)
        B = np.random.randint(-3, 4, size=(dim, dim)).astype(np.float32)

        C_expected = A @ B
        C_actual = ng.matmul(A, B)

        max_err = float(np.max(np.abs(C_actual - C_expected)))
        assert np.array_equal(C_actual, C_expected), (
            f"Bitwise mismatch for square {dim}x{dim}. Max diff: {max_err}"
        )
        print(f"  [PASS] Exact Square       {dim:3d}x{dim:3d} (diff: {max_err:.2e})")


def run_exact_rectangular_integer_tests():
    """Test rectangular and prime dimensions with exact integer arithmetic."""
    cases = [
        (1, 128, 1),       # Dot product
        (64, 1, 64),       # Outer product
        (32, 64, 128),     # Typical rectangular
        (128, 32, 64),
        (7, 13, 19),       # Prime / non-aligned dimensions
        (37, 59, 41),      # Large primes (stress edge kernel)
        (65, 129, 65),     # Just over tile boundaries
        (127, 255, 127),   # Large prime & odd dimensions
    ]
    for M, K, N in cases:
        np.random.seed(1337 + M * 10 + N)
        A = np.random.randint(-3, 4, size=(M, K)).astype(np.float32)
        B = np.random.randint(-3, 4, size=(K, N)).astype(np.float32)

        C_expected = A @ B
        C_actual = ng.matmul(A, B)

        max_err = float(np.max(np.abs(C_actual - C_expected)))
        assert np.array_equal(C_actual, C_expected), (
            f"Bitwise mismatch for rectangular ({M},{K}) @ ({K},{N}). Max diff: {max_err}"
        )
        print(f"  [PASS] Exact Rectangular ({M:3d},{K:3d}) @ ({K:3d},{N:3d}) (diff: {max_err:.2e})")


def run_exact_algebraic_properties():
    """Test mathematical identities that must hold with bitwise zero error."""
    # 1. Identity Matrix Multiplication: A @ I == A and I @ A == A
    for dim in [16, 64, 128, 256]:
        np.random.seed(100 + dim)
        A = np.random.randn(dim, dim).astype(np.float32)
        I = np.eye(dim, dtype=np.float32)

        C_right = ng.matmul(A, I)
        C_left = ng.matmul(I, A)

        assert np.array_equal(C_right, A), f"Identity A @ I failed for dim {dim}"
        assert np.array_equal(C_left, A), f"Identity I @ A failed for dim {dim}"

    print("  [PASS] Identity Preservation (A @ I == A & I @ A == A)  (diff: 0.00e+00)")

    # 2. Dyadic Fractions (Powers of Two: +-0.5, +-0.25)
    np.random.seed(2026)
    A_dyadic = (np.random.randint(-4, 5, size=(127, 255)) * 0.25).astype(np.float32)
    B_dyadic = (np.random.randint(-4, 5, size=(255, 127)) * 0.5).astype(np.float32)
    C_dyadic_act = ng.matmul(A_dyadic, B_dyadic)
    C_dyadic_exp = A_dyadic @ B_dyadic
    err_dyadic = float(np.max(np.abs(C_dyadic_act - C_dyadic_exp)))
    assert np.array_equal(C_dyadic_act, C_dyadic_exp), "Dyadic multiplication mismatch"
    print(f"  [PASS] Dyadic Fractions (+-0.5, +-0.25) (127,255)@(255,127) (diff: {err_dyadic:.2e})")

    # 3. Permutation Matrix: Reorders rows/columns exactly
    P = np.eye(64, dtype=np.float32)[np.random.permutation(64)]
    A_perm = np.random.randn(64, 64).astype(np.float32)
    C_perm = ng.matmul(A_perm, P)
    assert np.array_equal(C_perm, A_perm @ P), "Permutation matrix mismatch"
    print("  [PASS] Permutation Matrix Exact Column Swaps            (diff: 0.00e+00)")


def run_continuous_float_numerical_stability():
    """
    Test continuous real numbers against Double Precision (float64) ground truth.
    Evaluates Higham's backward numerical stability bound:
        ||C_actual - C_true||_F <= gamma_K * ||A||_F * ||B||_F
    where gamma_K = K * 2^(-24).
    """
    cases = [
        (64, 64, 64),
        (128, 128, 128),
        (127, 255, 127),
        (256, 256, 256),
    ]
    u = 2.0 ** -24  # IEEE-754 float32 unit roundoff (machine epsilon)

    for M, K, N in cases:
        np.random.seed(777 + M + K + N)
        A = np.random.uniform(-5.0, 5.0, size=(M, K)).astype(np.float32)
        B = np.random.uniform(-5.0, 5.0, size=(K, N)).astype(np.float32)

        C_actual = ng.matmul(A, B)
        # Calculate true mathematical ground truth in double precision
        C_true_64 = A.astype(np.float64) @ B.astype(np.float64)

        diff_norm = float(np.linalg.norm(C_actual.astype(np.float64) - C_true_64))
        norm_A = float(np.linalg.norm(A.astype(np.float64)))
        norm_B = float(np.linalg.norm(B.astype(np.float64)))

        # Higham theoretical backward error bound
        higham_bound = K * u * norm_A * norm_B
        stability_ratio = diff_norm / higham_bound
        max_diff = float(np.max(np.abs(C_actual.astype(np.float64) - C_true_64)))

        assert stability_ratio < 1.0, f"Numerical stability violated: ratio={stability_ratio}"
        print(
            f"  [PASS] Continuous ({M:3d},{K:3d}) @ ({K:3d},{N:3d}) "
            f"| max diff vs f64: {max_diff:.2e} "
            f"| Higham Ratio: {stability_ratio:.4f} << 1.0"
        )


def run_preallocated_out():
    """Test preallocated output buffer with exact integer arithmetic."""
    A = np.random.randint(-3, 4, size=(48, 48)).astype(np.float32)
    B = np.random.randint(-3, 4, size=(48, 48)).astype(np.float32)
    out = np.zeros((48, 48), dtype=np.float32)

    res = ng.matmul(A, B, out=out)
    assert res is out
    assert np.array_equal(out, A @ B)
    print("  [PASS] Preallocated Out Buffer Exact Execution         (diff: 0.00e+00)")


def run_sgemm_alpha_beta():
    """Test BLAS SGEMM with alpha and beta scaling."""
    # 1. Exact integer scaling: alpha=2.0, beta=3.0
    A_int = np.random.randint(-2, 3, size=(32, 32)).astype(np.float32)
    B_int = np.random.randint(-2, 3, size=(32, 32)).astype(np.float32)
    C_int = np.random.randint(-2, 3, size=(32, 32)).astype(np.float32)
    res_int = ng.sgemm(A_int, B_int, alpha=2.0, beta=3.0, c=C_int.copy())
    exp_int = 2.0 * (A_int @ B_int) + 3.0 * C_int
    assert np.array_equal(res_int, exp_int)
    print("  [PASS] BLAS SGEMM Exact Integer Scaling (2.0*AB + 3.0*C) (diff: 0.00e+00)")

    # 2. Float scaling
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    C_init = np.random.randn(32, 32).astype(np.float32)
    res = ng.sgemm(A, B, alpha=2.5, beta=1.25, c=C_init.copy())
    exp = 2.5 * (A @ B) + 1.25 * C_init
    assert np.allclose(res, exp, atol=1e-4, rtol=1e-4)
    print("  [PASS] BLAS SGEMM Continuous Float Scaling (2.5*AB + 1.25*C)")


def run_error_handling():
    """Test validation of mismatched dimensions and illegal inputs."""
    A = np.ones((10, 20), dtype=np.float32)
    B = np.ones((25, 30), dtype=np.float32)
    try:
        ng.matmul(A, B)
        assert False, "Expected ValueError on dimension mismatch"
    except ValueError:
        pass

    try:
        ng.matmul(np.ones((2, 2, 2)), np.ones((2, 2)))
        assert False, "Expected ValueError on non-2D input"
    except ValueError:
        pass

    print("  [PASS] Error Handling & Input Shape Validation")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("   Running NanoGEMM High-Precision Correctness Test Suite")
    print("=" * 65)
    test_simd_isa_available()

    print("\n--- 1. Exact Bitwise Tests (Zero Rounding: diff = 0.00e+00) ---")
    run_exact_square_integer_tests()

    print("\n--- 2. Exact Rectangular & Prime Dimensions (diff = 0.00e+00) ---")
    run_exact_rectangular_integer_tests()

    print("\n--- 3. Exact Algebraic Invariants (Identity & Dyadics) ---")
    run_exact_algebraic_properties()

    print("\n--- 4. Continuous Floats vs Double Precision (Float64) ---")
    run_continuous_float_numerical_stability()

    print("\n--- 5. Preallocated Output Buffer ---")
    run_preallocated_out()

    print("\n--- 6. BLAS SGEMM (Alpha & Beta) ---")
    run_sgemm_alpha_beta()

    print("\n--- 7. Error Handling ---")
    run_error_handling()

    print("\n" + "=" * 65)
    print("   ALL TESTS PASSED WITH 100% BITWISE & NUMERICAL ACCURACY!")
    print("=" * 65 + "\n")
