"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
"""

from nanogemm.core import (
    matmul,
    sgemm,
    bmm,
    matmul_int8,
    quantized_matmul,
    get_simd_isa,
    set_backend,
    get_backend,
)

__version__ = "0.3.5"
__all__ = [
    "matmul",
    "sgemm",
    "bmm",
    "matmul_int8",
    "quantized_matmul",
    "get_simd_isa",
    "set_backend",
    "get_backend",
    "__version__",
]
