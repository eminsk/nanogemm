"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
"""

from nanogemm.core import matmul, sgemm, get_simd_isa, set_backend, get_backend

__version__ = "0.2.0"
__all__ = ["matmul", "sgemm", "get_simd_isa", "set_backend", "get_backend", "__version__"]
