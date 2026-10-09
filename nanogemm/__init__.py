"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
"""

__version__ = "0.3.13"

from nanogemm.core import (
    matmul,
    sgemm,
    bmm,
    matmul_int8,
    quantized_matmul,
    get_simd_isa,
    set_backend,
    get_backend,
    score_batch,
    matmul_flat,
    is_available,
)
from nanogemm.mcp_server import NanoGEMMMCPServer

__all__ = [
    "matmul",
    "sgemm",
    "bmm",
    "matmul_int8",
    "quantized_matmul",
    "get_simd_isa",
    "set_backend",
    "get_backend",
    "score_batch",
    "matmul_flat",
    "is_available",
    "NanoGEMMMCPServer",
    "__version__",
]

