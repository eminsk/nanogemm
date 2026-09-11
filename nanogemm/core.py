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
# Pure Flat Assembler (FASM) Native DLL Loader
# ---------------------------------------------------------------------------

def _find_fasm_library() -> Optional[str]:
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    
    candidates = []
    if sys.platform.startswith("win"):
        is_64bit = sys.maxsize > 2**32
        dll_name = "nanogemm64.dll" if is_64bit else "nanogemm32.dll"
        candidates = [
            root_dir / "asm" / dll_name,
            pkg_dir / "asm" / dll_name,
            pkg_dir / dll_name,
            root_dir / "build" / dll_name,
            root_dir / dll_name,
        ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None

_fasm_path = _find_fasm_library()
_fasm_lib = None
if _fasm_path:
    try:
        _fasm_lib = ctypes.CDLL(_fasm_path)
        _fasm_lib.nanogemm_matmul.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _fasm_lib.nanogemm_matmul.restype = None

        _fasm_lib.nanogemm_sgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p, ctypes.c_int,
        ]
        _fasm_lib.nanogemm_sgemm.restype = None

        _fasm_lib.nanogemm_simd_isa.argtypes = []
        _fasm_lib.nanogemm_simd_isa.restype = ctypes.c_char_p
    except Exception:
        _fasm_lib = None

# ---------------------------------------------------------------------------
# Fallback: General Dynamic Library (ctypes) Loader
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
# Backend Management
# ---------------------------------------------------------------------------

# Default backend: "auto" selects FASM for large matrices (up to 2x faster)
# and C extension for small matrices (lower Python call overhead).
_ACTIVE_BACKEND = "auto" if (_fasm_lib is not None or _HAS_C_EXT) else "auto"

def set_backend(backend: str) -> None:
    """
    Set the execution backend for NanoGEMM.

    Parameters
    ----------
    backend : str
        One of:
        - 'fasm': Pure Flat Assembler x86/x64 microkernel (fastest on medium/large tensors)
        - 'c': Compiled C extension with AVX2 intrinsics
        - 'auto': Automatically select optimal backend based on tensor dimensions
    """
    global _ACTIVE_BACKEND
    b = backend.lower().strip()
    if b not in ("fasm", "c", "auto"):
        raise ValueError(f"Unknown backend '{backend}'. Choose from 'fasm', 'c', or 'auto'.")
    if b == "fasm" and _fasm_lib is None:
        raise RuntimeError("FASM native library not found. Compile it using asm/build.bat.")
    if b == "c" and not _HAS_C_EXT:
        raise RuntimeError("C extension is not available.")
    _ACTIVE_BACKEND = b

def get_backend() -> str:
    """Return the name of the currently active execution backend ('fasm', 'c', or 'auto')."""
    return _ACTIVE_BACKEND

# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------

def get_simd_isa() -> str:
    """Return the active SIMD instruction set in the compiled microkernel."""
    backend = _ACTIVE_BACKEND
    if backend == "fasm" and _fasm_lib is not None:
        raw = _fasm_lib.nanogemm_simd_isa()
        return raw.decode("utf-8") if raw else "FASM SIMD"
    if backend == "c" and _HAS_C_EXT:
        return _ext.simd_isa()
    if _HAS_C_EXT:
        return _ext.simd_isa()
    if _fasm_lib is not None:
        raw = _fasm_lib.nanogemm_simd_isa()
        return raw.decode("utf-8") if raw else "FASM SIMD"
    if _lib:
        raw = _lib.nanogemm_simd_isa()
        return raw.decode("utf-8") if raw else "Unknown"
    return "Not compiled"


def matmul(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
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
    backend : str, optional
        Override execution backend ('fasm', 'c', or 'auto').

    Returns
    -------
    np.ndarray
        Result of matrix multiplication (M, N) with float32 dtype.
    """
    if a.ndim > 2 or b.ndim > 2:
        return bmm(a, b, out=out)

    if backend is None and _HAS_C_EXT:
        try:
            if out is not None:
                _ext.matmul_fast(a, b, out)
                return out
            res_out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
            _ext.matmul_fast(a, b, res_out)
            return res_out
        except (TypeError, ValueError):
            pass

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

    chosen_backend = backend.lower().strip() if backend else _ACTIVE_BACKEND

    if chosen_backend == "auto":
        # For small matrices (<= 32x32), C extension has lower Python call overhead;
        # for medium/large matrices (>= 64x64), FASM assembly kernel is up to 2x faster.
        if M <= 32 and N <= 32 and _HAS_C_EXT:
            chosen_backend = "c"
        elif _fasm_lib is not None:
            chosen_backend = "fasm"
        elif _HAS_C_EXT:
            chosen_backend = "c"
        else:
            chosen_backend = "c"

    if chosen_backend == "fasm" and _fasm_lib is not None:
        _fasm_lib.nanogemm_matmul(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

    if chosen_backend == "c" and _HAS_C_EXT:
        _ext.matmul_fast(a, b, out)
        return out

    if _fasm_lib is not None:
        _fasm_lib.nanogemm_matmul(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

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
    backend: Optional[str] = None,
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

    chosen_backend = backend.lower().strip() if backend else _ACTIVE_BACKEND

    if chosen_backend == "auto":
        if M <= 32 and N <= 32 and _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
            chosen_backend = "c"
        elif _fasm_lib is not None:
            chosen_backend = "fasm"
        elif _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
            chosen_backend = "c"
        else:
            chosen_backend = "c"

    if chosen_backend == "fasm" and _fasm_lib is not None:
        _fasm_lib.nanogemm_sgemm(
            M, N, K,
            float(alpha),
            a.ctypes.data, K,
            b.ctypes.data, N,
            float(beta),
            c.ctypes.data, N
        )
        return c

    if chosen_backend == "c" and _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
        _ext.sgemm_fast(a, b, float(alpha), float(beta), c)
        return c

    if _fasm_lib is not None:
        _fasm_lib.nanogemm_sgemm(
            M, N, K,
            float(alpha),
            a.ctypes.data, K,
            b.ctypes.data, N,
            float(beta),
            c.ctypes.data, N
        )
        return c

    if _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
        _ext.sgemm_fast(a, b, float(alpha), float(beta), c)
        return c

    if _lib:
        _lib.nanogemm_sgemm(M, N, K, float(alpha), a.ctypes.data, K, b.ctypes.data, N, float(beta), c.ctypes.data, N)
        return c

    # Fallback to matmul
    res = matmul(a, b, backend=chosen_backend)
    if beta == 0.0:
        np.multiply(res, alpha, out=c)
    else:
        c[:] = alpha * res + beta * c
    return c


def bmm(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Batched Matrix Multiplication (BMM) for 3D and 4D tensors with hardware acceleration.

    Supports:
    - 3D x 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
    - 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N) (broadcasting A across batch)
    - 3D x 2D: (B, M, K) @ (K, N) -> (B, M, N) (broadcasting B across batch)
    - 4D x 4D: (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N) (e.g. Transformer Multi-Head Attention)

    Parameters
    ----------
    a : np.ndarray
        Input tensor (2D, 3D, or 4D).
    b : np.ndarray
        Input tensor (2D, 3D, or 4D).
    out : np.ndarray, optional
        Pre-allocated output buffer matching the output shape and dtype float32.

    Returns
    -------
    np.ndarray
        Result of batched matrix multiplication.
    """
    orig_shape_a = a.shape
    orig_shape_b = b.shape

    # Handle 4D Multi-Head Attention tensors: (B, H, M, K) -> flatten to 3D (B*H, M, K)
    is_4d = False
    out_shape_4d = None
    if a.ndim == 4 and b.ndim == 4:
        if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
            raise ValueError(f"Leading batch/head dimensions mismatch: {orig_shape_a} vs {orig_shape_b}")
        B_dim, H_dim, M, K = a.shape
        _, _, K_b, N = b.shape
        if K != K_b:
            raise ValueError(f"Inner dimension mismatch: {K} vs {K_b}")
        a = a.reshape(B_dim * H_dim, M, K)
        b = b.reshape(B_dim * H_dim, K, N)
        out_shape_4d = (B_dim, H_dim, M, N)
        is_4d = True

    if not a.flags.c_contiguous or a.dtype != np.float32:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous or b.dtype != np.float32:
        b = np.ascontiguousarray(b, dtype=np.float32)

    # Compute expected 3D output shape
    if a.ndim == 3 and b.ndim == 3:
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"Batch dimension mismatch: {a.shape[0]} vs {b.shape[0]}")
        out_shape_3d = (a.shape[0], a.shape[1], b.shape[2])
    elif a.ndim == 2 and b.ndim == 3:
        out_shape_3d = (b.shape[0], a.shape[0], b.shape[2])
    elif a.ndim == 3 and b.ndim == 2:
        out_shape_3d = (a.shape[0], a.shape[1], b.shape[1])
    else:
        raise ValueError(f"Unsupported tensor shapes for bmm: {orig_shape_a} and {orig_shape_b}")

    target_shape = out_shape_4d if is_4d else out_shape_3d

    if out is None:
        out_buf = np.empty(out_shape_3d, dtype=np.float32)
    else:
        if out.shape != target_shape or out.dtype != np.float32:
            raise ValueError(f"out must be float32 array with shape {target_shape}")
        out_buf = out.reshape(out_shape_3d) if is_4d else out
        if not out_buf.flags.c_contiguous:
            out_buf = np.ascontiguousarray(out_buf)

    if _HAS_C_EXT and hasattr(_ext, "bmm_fast"):
        _ext.bmm_fast(a, b, out_buf)
    else:
        batch_count = out_shape_3d[0]
        for i in range(batch_count):
            a_slice = a[i] if a.ndim == 3 else a
            b_slice = b[i] if b.ndim == 3 else b
            out_buf[i] = matmul(a_slice, b_slice)

    if is_4d:
        return out_buf.reshape(out_shape_4d)
    return out_buf


def matmul_int8(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Multiply two 2D matrices of signed 8-bit integers (int8) with hardware SIMD acceleration.
    Accumulates in 32-bit signed integers (int32) to prevent numeric overflow.

    Parameters
    ----------
    a : np.ndarray
        Matrix of shape (M, K) and dtype int8.
    b : np.ndarray
        Matrix of shape (K, N) and dtype int8.
    out : np.ndarray, optional
        Pre-allocated output buffer of shape (M, N) and dtype int32.

    Returns
    -------
    np.ndarray
        Result of matrix multiplication with int32 dtype and shape (M, N).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays for matmul_int8, got a.ndim={a.ndim} and b.ndim={b.ndim}")

    M, K = a.shape
    K_b, N = b.shape
    if K != K_b:
        raise ValueError(f"Dimension mismatch in matmul_int8: ({M}, {K}) x ({K_b}, {N})")

    if not a.flags.c_contiguous or a.dtype != np.int8:
        a = np.ascontiguousarray(a, dtype=np.int8)
    if not b.flags.c_contiguous or b.dtype != np.int8:
        b = np.ascontiguousarray(b, dtype=np.int8)

    if out is None:
        out = np.empty((M, N), dtype=np.int32)
    else:
        if out.shape != (M, N) or out.dtype != np.int32 or not out.flags.c_contiguous:
            raise ValueError(f"out must be contiguous int32 array of shape ({M}, {N})")

    if _HAS_C_EXT and hasattr(_ext, "matmul_int8_fast"):
        _ext.matmul_int8_fast(a, b, out)
        return out

    # Fallback
    np.matmul(a.astype(np.int64), b.astype(np.int64), out=out)
    return out


def quantized_matmul(
    a: np.ndarray,
    b: np.ndarray,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    bias: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Quantized linear layer forward pass: Y = (A_int8 @ B_int8) * (scale_a * scale_b) + bias.
    Computes integer GEMM with AVX2 SIMD and returns dequantized float32 output.

    Parameters
    ----------
    a : np.ndarray
        Quantized input activations (int8).
    b : np.ndarray
        Quantized weights (int8).
    scale_a : float, default 1.0
        Dequantization scale for matrix A.
    scale_b : float, default 1.0
        Dequantization scale for matrix B.
    bias : np.ndarray, optional
        Optional bias vector/matrix to add to the output.
    out : np.ndarray, optional
        Pre-allocated float32 output buffer.

    Returns
    -------
    np.ndarray
        Dequantized float32 output matrix.
    """
    int_accum = matmul_int8(a, b)
    effective_scale = np.float32(scale_a * scale_b)

    if out is None:
        res = int_accum.astype(np.float32) * effective_scale
    else:
        if out.shape != int_accum.shape or out.dtype != np.float32:
            raise ValueError(f"out must be float32 array with shape {int_accum.shape}")
        np.multiply(int_accum, effective_scale, out=out, dtype=np.float32)
        res = out

    if bias is not None:
        res += bias

    return res
