"""
Benchmark Suite: Batched Matrix Multiplication (BMM) & Quantized INT8 in NanoGEMM.
"""

import time
import numpy as np
import nanogemm as ng

print("\n" + "=" * 80)
print(f"  NanoGEMM Batched & Quantized Benchmark Suite")
print(f"  Active SIMD ISA: {ng.get_simd_isa()}")
print("=" * 80)

# -------------------------------------------------------------
# 1. Batched Multi-Head Attention GEMM (B x H x S x D)
# -------------------------------------------------------------
print("\n[1] Multi-Head Attention Batched GEMM (Q @ K.T)")
print("-" * 80)
print(f"{'Attention Shape':<25} | {'NumPy (µs)':<12} | {'NanoGEMM (µs)':<14} | {'Speedup':<10}")
print("-" * 80)

mha_configs = [
    ("Heads=8, Seq=32, Dim=32", 8, 32, 32, 32),
    ("Heads=16, Seq=32, Dim=64", 16, 32, 64, 32),
    ("Heads=32, Seq=64, Dim=64", 32, 64, 64, 64),
    ("Heads=64, Seq=32, Dim=32", 64, 32, 32, 32),
]

for label, heads, seq, dim, seq_out in mha_configs:
    Q = np.random.randn(heads, seq, dim).astype(np.float32)
    K = np.random.randn(heads, dim, seq_out).astype(np.float32)
    C_out = np.empty((heads, seq, seq_out), dtype=np.float32)

    # Warmup
    for _ in range(50):
        _ = Q @ K
        ng.bmm(Q, K, out=C_out)

    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = Q @ K
    t_np = (time.perf_counter() - t0) / iters * 1e6

    t0 = time.perf_counter()
    for _ in range(iters):
        ng.bmm(Q, K, out=C_out)
    t_ng = (time.perf_counter() - t0) / iters * 1e6

    speedup = t_np / t_ng
    print(f"{label:<25} | {t_np:>10.2f} µs | {t_ng:>12.2f} µs | {speedup:>8.2f}x (FASTER)")

# -------------------------------------------------------------
# 2. Quantized INT8 GEMM Throughput (vs Float32)
# -------------------------------------------------------------
print("\n[2] Quantized INT8 GEMM: NanoGEMM vs NumPy (int8 x int8 -> int32)")
print("-" * 80)
print(f"{'Matrix Size':<15} | {'NumPy INT8 (µs)':<16} | {'NanoGEMM INT8 (µs)':<18} | {'Speedup':<10} | {'Throughput':<12}")
print("-" * 80)

int8_sizes = [16, 32, 64, 128]
for dim in int8_sizes:
    A_i8 = np.random.randint(-128, 128, size=(dim, dim), dtype=np.int8)
    B_i8 = np.random.randint(-128, 128, size=(dim, dim), dtype=np.int8)
    C_i32 = np.empty((dim, dim), dtype=np.int32)

    # Warmup
    for _ in range(50):
        _ = A_i8 @ B_i8
        ng.matmul_int8(A_i8, B_i8, out=C_i32)

    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = A_i8 @ B_i8
    t_np_i8 = (time.perf_counter() - t0) / iters * 1e6

    t0 = time.perf_counter()
    for _ in range(iters):
        ng.matmul_int8(A_i8, B_i8, out=C_i32)
    t_ng_i8 = (time.perf_counter() - t0) / iters * 1e6

    ops = 2.0 * (dim ** 3)
    gops_i8 = (ops / (t_ng_i8 * 1e-6)) / 1e9
    speedup = t_np_i8 / t_ng_i8

    print(f"{dim}x{dim:<12} | {t_np_i8:>14.2f} µs | {t_ng_i8:>16.2f} µs | {speedup:>8.2f}x | {gops_i8:>8.2f} GOP/s")

print("=" * 80 + "\n")
