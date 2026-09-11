"""
Comprehensive Test Suite for NanoGEMM BMM and Quantized INT8 Extensions.
Tests numerical correctness, broadcasting, 4D attention shapes, and bit-for-bit INT8 precision.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import nanogemm as ng
except ImportError:
    for p in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
        if (p / "nanogemm" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            break
    import nanogemm as ng


def test_bmm_3d_square():
    rng = np.random.default_rng(42)
    for B, dim in [(1, 16), (4, 24), (8, 32), (2, 64)]:
        A = rng.standard_normal((B, dim, dim)).astype(np.float32)
        B_mat = rng.standard_normal((B, dim, dim)).astype(np.float32)
        C_ref = A @ B_mat

        # Test bmm
        C_ng = ng.bmm(A, B_mat)
        assert np.allclose(C_ng, C_ref, atol=1e-4), f"BMM 3D failed for shape ({B}, {dim}, {dim})"

        # Test matmul automatic routing
        C_matmul = ng.matmul(A, B_mat)
        assert np.allclose(C_matmul, C_ref, atol=1e-4), f"matmul 3D routing failed for shape ({B}, {dim}, {dim})"


def test_bmm_3d_non_square():
    rng = np.random.default_rng(123)
    B, M, K, N = 3, 23, 47, 31
    A = rng.standard_normal((B, M, K)).astype(np.float32)
    B_mat = rng.standard_normal((B, K, N)).astype(np.float32)
    C_ref = A @ B_mat

    C_ng = ng.bmm(A, B_mat)
    assert np.allclose(C_ng, C_ref, atol=1e-4), "BMM 3D non-square failed"


def test_bmm_broadcasting():
    rng = np.random.default_rng(456)
    B, M, K, N = 4, 16, 32, 24
    A_2d = rng.standard_normal((M, K)).astype(np.float32)
    B_3d = rng.standard_normal((B, K, N)).astype(np.float32)
    C_ref1 = A_2d @ B_3d
    C_ng1 = ng.bmm(A_2d, B_3d)
    assert np.allclose(C_ng1, C_ref1, atol=1e-4), "BMM 2Dx3D broadcasting failed"

    A_3d = rng.standard_normal((B, M, K)).astype(np.float32)
    B_2d = rng.standard_normal((K, N)).astype(np.float32)
    C_ref2 = A_3d @ B_2d
    C_ng2 = ng.bmm(A_3d, B_2d)
    assert np.allclose(C_ng2, C_ref2, atol=1e-4), "BMM 3Dx2D broadcasting failed"


def test_bmm_4d_multi_head_attention():
    rng = np.random.default_rng(789)
    # Batch=2, Heads=4, SeqLen=16, HeadDim=32
    B, H, S, D = 2, 4, 16, 32
    Q = rng.standard_normal((B, H, S, D)).astype(np.float32)
    K = rng.standard_normal((B, H, D, S)).astype(np.float32)
    Scores_ref = Q @ K
    Scores_ng = ng.bmm(Q, K)
    assert np.allclose(Scores_ng, Scores_ref, atol=1e-4), "4D Multi-Head Attention BMM failed"
    assert Scores_ng.shape == (B, H, S, S), f"Unexpected shape {Scores_ng.shape}"


def test_bmm_preallocated_buffer():
    rng = np.random.default_rng(101)
    B, M, K, N = 5, 24, 24, 24
    A = rng.standard_normal((B, M, K)).astype(np.float32)
    B_mat = rng.standard_normal((B, K, N)).astype(np.float32)
    out_buf = np.zeros((B, M, N), dtype=np.float32)

    res = ng.bmm(A, B_mat, out=out_buf)
    assert res is out_buf
    assert np.allclose(out_buf, A @ B_mat, atol=1e-4)


def test_matmul_int8_square():
    rng = np.random.default_rng(202)
    for dim in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
        A = rng.integers(-128, 128, size=(dim, dim), dtype=np.int8)
        B = rng.integers(-128, 128, size=(dim, dim), dtype=np.int8)
        C_ref = A.astype(np.int64) @ B.astype(np.int64)

        C_ng = ng.matmul_int8(A, B)
        assert C_ng.dtype == np.int32
        assert np.array_equal(C_ng, C_ref), f"INT8 GEMM failed for size {dim}x{dim}"


def test_matmul_int8_non_square():
    rng = np.random.default_rng(303)
    for (M, K, N) in [(37, 53, 29), (1, 128, 1), (17, 31, 13), (64, 32, 128)]:
        A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        C_ref = A.astype(np.int64) @ B.astype(np.int64)

        C_ng = ng.matmul_int8(A, B)
        assert C_ng.dtype == np.int32
        assert np.array_equal(C_ng, C_ref), f"INT8 GEMM non-square failed for shape ({M},{K})x({K},{N})"


def test_matmul_int8_preallocated():
    rng = np.random.default_rng(404)
    A = rng.integers(-128, 128, size=(32, 32), dtype=np.int8)
    B = rng.integers(-128, 128, size=(32, 32), dtype=np.int8)
    out = np.empty((32, 32), dtype=np.int32)

    res = ng.matmul_int8(A, B, out=out)
    assert res is out
    assert np.array_equal(out, A.astype(np.int64) @ B.astype(np.int64))


def test_quantized_matmul():
    rng = np.random.default_rng(505)
    M, K, N = 32, 64, 32
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    scale_a = 0.05
    scale_b = 0.02
    bias = rng.standard_normal(N).astype(np.float32)

    # Reference float calculation
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)) * (scale_a * scale_b) + bias

    # NanoGEMM quantized forward
    C_ng = ng.quantized_matmul(A, B, scale_a=scale_a, scale_b=scale_b, bias=bias)
    assert C_ng.dtype == np.float32
    assert np.allclose(C_ng, C_ref, atol=1e-3), "quantized_matmul calculation mismatch!"


if __name__ == "__main__":
    test_bmm_3d_square()
    test_bmm_3d_non_square()
    test_bmm_broadcasting()
    test_bmm_4d_multi_head_attention()
    test_bmm_preallocated_buffer()
    test_matmul_int8_square()
    test_matmul_int8_non_square()
    test_matmul_int8_preallocated()
    test_quantized_matmul()
    print("ALL BMM & INT8 TESTS PASSED WITH 100% NUMERICAL PRECISION!")
