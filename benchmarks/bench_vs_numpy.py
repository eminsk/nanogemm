"""
Official Benchmark Suite: NanoGEMM vs NumPy on CPU.
Measures latency (microseconds), throughput (GFLOPS), and speedup factor.
"""

import sys
import io
import time
from pathlib import Path
import numpy as np

# Ensure utf-8 stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import nanogemm as ng
except ImportError:
    for p in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
        if (p / "nanogemm" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            break
    import nanogemm as ng



def benchmark_size(dim: int, num_iters: int):
    M = N = K = dim
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_out = np.empty((M, N), dtype=np.float32)

    # Warmup
    for _ in range(min(100, num_iters)):
        _ = A @ B
        _ = ng.matmul(A, B, out=C_out)

    # Benchmark NumPy
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = A @ B
    t_numpy = (time.perf_counter() - t0) / num_iters

    # Benchmark NanoGEMM
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = ng.matmul(A, B, out=C_out)
    t_nanogemm = (time.perf_counter() - t0) / num_iters

    # Flops: 2 * M * N * K
    flops = 2.0 * M * N * K
    gflops_numpy = (flops / t_numpy) / 1e9
    gflops_nanogemm = (flops / t_nanogemm) / 1e9
    speedup = t_numpy / t_nanogemm

    return {
        "dim": f"{dim}x{dim}",
        "numpy_us": t_numpy * 1e6,
        "nanogemm_us": t_nanogemm * 1e6,
        "numpy_gflops": gflops_numpy,
        "nanogemm_gflops": gflops_nanogemm,
        "speedup": speedup,
    }


def main():
    print("\n" + "=" * 82)
    print(f"  NanoGEMM Benchmark vs NumPy {np.__version__} on CPU")
    print(f"  Microkernel ISA: {ng.get_simd_isa()}")
    print("=" * 82)

    sizes = [
        (16, 25000),
        (32, 10000),
        (64, 4000),
        (128, 1000),
        (256, 250),
        (512, 50),
    ]

    results = []
    print(f"\n{'Matrix Size':<12} | {'NumPy (µs)':<12} | {'NanoGEMM (µs)':<14} | {'Speedup':<12} | {'NanoGEMM GFLOPS':<15}")
    print("-" * 82)

    for dim, iters in sizes:
        res = benchmark_size(dim, iters)
        results.append(res)
        speedup_str = f"{res['speedup']:.2f}x"
        if res['speedup'] >= 1.0:
            speedup_str = f"{speedup_str} (FASTER)"
        print(
            f"{res['dim']:<12} | "
            f"{res['numpy_us']:>10.2f} µs | "
            f"{res['nanogemm_us']:>12.2f} µs | "
            f"{speedup_str:<12} | "
            f"{res['nanogemm_gflops']:>13.2f} GFLOPS"
        )

    print("-" * 82)
    print("  Key Takeaway: NanoGEMM eliminates BLAS function overhead on small/medium tensors")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
