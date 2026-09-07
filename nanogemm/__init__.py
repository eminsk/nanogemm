"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
"""

from nanogemm.core import matmul, sgemm, get_simd_isa

__version__ = "0.2.0"
__all__ = ["matmul", "sgemm", "get_simd_isa", "__version__"]
