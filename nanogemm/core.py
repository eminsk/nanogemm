"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
Zero-overhead CPU microkernels for AI and scientific computing in Python.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Fast-Path: Native C-Extension Loader
# ---------------------------------------------------------------------------

_HAS_C_EXT = False
try:
    from nanogemm import _ext  # type: ignore
    _HAS_C_EXT = True
except ImportError:
    try:
        import _ext  # type: ignore
        _HAS_C_EXT = True
    except ImportError:
        _HAS_C_EXT = False

# ---------------------------------------------------------------------------
# Fallback: Dynamic Library (ctypes) Loader
# ---------------------------------------------------------------------------

def _find_library() -> Optional[str]:
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    
    candidates = []
    if sys.platform.startswith("win"):
        dll_name = "nanogemm.dll"
        candidates = [
            pkg_dir / dll_name,
            root_dir / "build" / dll_name,
            root_dir / dll_name,
        ]
    elif sys.platform == "darwin":
        dylib_name = "libnanogemm.dylib"
        candidates = [
            pkg_dir / dylib_name,
            root_dir / "build" / dylib_name,
            root_dir / dylib_name,
        ]
    else:
        so_name = "libnanogemm.so"
        candidates = [
            pkg_dir / so_name,
            root_dir / "build" / so_name,
            root_dir / so_name,
        ]

    for p in candidates:
        if p.is_file():
            return str(p)
    return None

_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path) if _lib_path else None

if _lib:
    _c_float_p = ctypes.POINTER(ctypes.c_float)
    _lib.nanogemm_sgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
    ]
    _lib.nanogemm_sgemm.restype = None

    _lib.nanogemm_matmul.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _lib.nanogemm_matmul.restype = None

    _lib.nanogemm_simd_isa.argtypes = []
    _lib.nanogemm_simd_isa.restype = ctypes.c_char_p


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------

def get_simd_isa() -> str:
    """Return the active SIMD instruction set in the compiled microkernel."""
    if _HAS_C_EXT:
        return _ext.simd_isa()
    if _lib:
        raw = _lib.nanogemm_simd_isa()
        return raw.decode("utf-8") if raw else "Unknown"
    return "Not compiled"


def matmul(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Multiply two 2D matrices using hardware-accelerated SIMD GEMM.

    Parameters
    ----------
    a : np.ndarray
        Matrix of shape (M, K). Must be 2-dimensional.
    b : np.ndarray
        Matrix of shape (K, N). Must be 2-dimensional.
    out : np.ndarray, optional
        Pre-allocated output buffer of shape (M, N) and dtype float32.

    Returns
    -------
    np.ndarray
        Result of matrix multiplication (M, N) with float32 dtype.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got a.ndim={a.ndim} and b.ndim={b.ndim}"
        )

    M, K = a.shape
    K_b, N = b.shape

    if K != K_b:
        raise ValueError(
            f"Incompatible matrix dimensions: cannot multiply ({M}, {K}) by ({K_b}, {N})"
        )

    if not a.flags.c_contiguous or a.dtype != np.float32:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous or b.dtype != np.float32:
        b = np.ascontiguousarray(b, dtype=np.float32)

    if out is None:
        out = np.empty((M, N), dtype=np.float32)
    else:
        if out.shape != (M, N) or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError(
                f"out must be contiguous float32 array of shape ({M}, {N})"
            )

    if _HAS_C_EXT:
        _ext.matmul_fast(a, b, out)
        return out

    if _lib:
        _lib.nanogemm_matmul(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

    raise RuntimeError("NanoGEMM binary not available. Please compile nanogemm native binary.")


def sgemm(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Standard BLAS SGEMM: C = alpha * (A @ B) + beta * C.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {a.ndim} and {b.ndim}")

    M, K = a.shape
    K_b, N = b.shape
    if K != K_b:
        raise ValueError(f"Dimension mismatch: {a.shape} vs {b.shape}")

    if not a.flags.c_contiguous or a.dtype != np.float32:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous or b.dtype != np.float32:
        b = np.ascontiguousarray(b, dtype=np.float32)

    if c is None:
        c = np.zeros((M, N), dtype=np.float32)
    else:
        if c.shape != (M, N) or c.dtype != np.float32 or not c.flags.c_contiguous:
            raise ValueError(f"c must be contiguous float32 array of shape ({M}, {N})")

    if _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
        _ext.sgemm_fast(a, b, float(alpha), float(beta), c)
        return c

    if _lib:
        _lib.nanogemm_sgemm(M, N, K, float(alpha), a.ctypes.data, K, b.ctypes.data, N, float(beta), c.ctypes.data, N)
        return c

    # Fallback to matmul
    res = matmul(a, b)
    if beta == 0.0:
        np.multiply(res, alpha, out=c)
    else:
        c[:] = alpha * res + beta * c
    return c
