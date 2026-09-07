# NanoGEMM ⚡

[![PyPI](https://img.shields.io/pypi/v/nanogemm.svg?color=blue)](https://pypi.org/project/nanogemm/)
[![CI](https://github.com/eminsk/nanogemm/actions/workflows/ci.yml/badge.svg)](https://github.com/eminsk/nanogemm/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/nanogemm/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA%20(256--bit)-brightgreen)](https://github.com/eminsk/nanogemm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eminsk/nanogemm/blob/main/notebooks/benchmark.ipynb)
[![Footprint](https://img.shields.io/badge/Binary-~100_KB-orange)](https://github.com/eminsk/nanogemm)
[![Dev.to](https://img.shields.io/badge/Dev.to-Read%20Article-0a0a0a?logo=devdotto)](https://dev.to/eminsk/how-i-beat-numpy-matrix-multiplication-by-28x-with-a-100kb-c-microkernel-82k)

**NanoGEMM** is a minimalist, bare-metal General Matrix Multiplication (GEMM) engine designed for sub-microsecond CPU inference and high-performance computing in Python.

Built with direct **AVX2 / FMA (256-bit SIMD)** assembly-level register tiling and cache blocking, NanoGEMM eliminates the heavy function-call dispatch, thread-pool barriers, and memory-packing overhead of heavyweight BLAS libraries (OpenBLAS, MKL) for small-to-medium tensors.

---

## 🚀 Performance Benchmarks

Measured on **Intel/AMD x86-64 CPU (AVX2 + FMA)** against **NumPy 2.2.3** (single-precision `float32`):

| Matrix Dimension | NumPy 2.2.3 Latency | NanoGEMM Latency | Speedup Factor | NanoGEMM Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **`16 x 16`** | `3.21 µs` | **`1.23 µs`** (C: `0.65 µs`) | 🚀 **2.83x FASTER** | `2.83 GFLOPS` |
| **`32 x 32`** | `5.75 µs` | **`2.74 µs`** (C: `2.18 µs`) | 🚀 **2.26x FASTER** | `15.13 GFLOPS` |
| **`64 x 64`** | `18.70 µs` | **`17.76 µs`** (C: `16.39 µs`) | 🚀 **1.10x FASTER** | `27.08 GFLOPS` |
| **`128 x 128`** | `114.07 µs` | `182.46 µs` | `0.60x` | `23.42 GFLOPS` |
| **`256 x 256`** | `426.24 µs` | `1501.24 µs` | `0.28x` | `22.35 GFLOPS` |

> 💡 **Why is NanoGEMM faster on small/medium matrices?**  
> Traditional BLAS engines incur 3–10 µs of fixed overhead per invocation due to dynamic runtime dispatch, argument sanitization, thread synchronization, and packing buffers. NanoGEMM utilizes a zero-allocation, direct register-tiled microkernel that executes in **sub-microsecond time** immediately upon invocation.

---

## 🛠 Architectural Design

### 1. Register Tiling ($6 \times 16$ Microkernel)
* **Register allocation:** Utilizes 12 `ymm` registers (`ymm0` – `ymm11`) as 256-bit floating-point accumulators storing a $6 \times 16$ tile of matrix $C$.
* **Vector broadcast & FMA:** Two `ymm` registers load vectors from $B$, while individual scalar elements of $A$ are broadcast across `ymm` using `_mm256_set1_ps` and accumulated via fused multiply-add (`_mm256_fmadd_ps`).
* **Zero Spilling:** Fits completely inside the 16 available x86-64 YMM registers without stack eviction.

### 2. Multi-Level Cache Blocking
* **$L_1$ / $L_2$ Cache Tiling:** Matrices are processed in cache blocks ($M_c = 64, N_c = 128, K_c = 128$) to maintain maximum L1/L2 data cache hit ratios and eliminate memory bus thrashing.
* **Vectorized Edge Handling:** Arbitrary matrix dimensions (non-multiples of 6 or 16) are processed using boundary SIMD edge loops without padding or buffer allocations.

```
       Matrix A (M x K)              Matrix B (K x N)
     [ . . . . . . . . ]           [ . . . ymm0 . . . ]
     [ . . . . . . . . ]           [ . . . ymm1 . . . ]
     [ a0 a1 a2 a3 . . ]     x     [ . . . . .  . . . ]
     [ . . . . . . . . ]           [ . . . . .  . . . ]
     [ . . . . . . . . ]
             │                             │
             └──────────────┬──────────────┘
                            ▼
                Matrix C (6 x 16 Tile)
             [ ymm0  ymm1  ] -> Row 0
             [ ymm2  ymm3  ] -> Row 1
             [ ymm4  ymm5  ] -> Row 2
             [ ymm6  ymm7  ] -> Row 3
             [ ymm8  ymm9  ] -> Row 4
             [ ymm10 ymm11 ] -> Row 5
```

---

## 📦 Installation & Quickstart

### Installation via PyPI (Recommended)
```bash
pip install nanogemm
```

### Build from Source
```bash
git clone https://github.com/eminsk/nanogemm.git
cd nanogemm
pip install -e .
```

### Python Usage
```python
import nanogemm as ng
import numpy as np

# Verify SIMD hardware acceleration
print("Active ISA:", ng.get_simd_isa())
# Output: Active ISA: AVX2+FMA (256-bit SIMD, 6x16 register tiling)

# Allocate input matrices
A = np.random.randn(32, 64).astype(np.float32)
B = np.random.randn(64, 128).astype(np.float32)

# Direct hardware-accelerated MatMul: C = A @ B
C = ng.matmul(A, B)

# Or with pre-allocated zero-copy output buffer for maximum performance:
out = np.empty((32, 128), dtype=np.float32)
ng.matmul(A, B, out=out)

# Standard BLAS SGEMM interface: C = alpha * (A @ B) + beta * C
res = ng.sgemm(A, B, alpha=2.0, beta=0.5, c=out)
```

---

## 🧪 Testing & Verification

Run the comprehensive correctness test suite comparing NanoGEMM with NumPy reference outputs across random uniforms, normals, non-square dimensions, and prime shapes:

```bash
python tests/test_correctness.py
```

Run the official benchmark against your installed NumPy BLAS:

```bash
python benchmarks/bench_vs_numpy.py
```

---

## 🌐 High-Performance Systems Ecosystem

NanoGEMM is developed by [**@eminsk**](https://github.com/eminsk) as part of an engineering ecosystem focused on low-level hardware performance, assembly programming, and native desktop computing:

* 🎥 [**screenvideo**](https://github.com/eminsk/screenvideo) — Lightweight desktop screen recorder featuring WASAPI loopback audio and a standalone pure x64 Flat Assembler (FASM) native edition.
* 📊 [**xlsx_vievers**](https://github.com/eminsk/xlsx_vievers) — Desktop spreadsheet processor with 80+ formula functions, Chart Wizard, and hardware-accelerated SIMD SSE2 math engine.
* 📈 [**yfinance-ta-patterns**](https://github.com/eminsk/yfinance-ta-patterns) — Candlestick pattern scanner and AI ranking suite powered by TA-Lib and quantitative backtesting.
* 🔍 [**StackOverflowAPI**](https://github.com/eminsk/StackOverflowAPI) — Desktop client for Stack Overflow built with CustomTkinter and native FASM x64 search client.

---

## 📄 License

MIT License — Copyright (c) 2026 [eminsk](https://github.com/eminsk).
