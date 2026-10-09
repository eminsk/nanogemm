"""
NanoGEMM: Minimalist, bare-metal SIMD & Assembly Matrix Multiplication Engine.
Zero-overhead CPU microkernels for AI and scientific computing in Python.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional, Any, Tuple, List, Union

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None

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
# Cross-Platform Native Library Loader & Symbol Binder
# Supports: Windows (FASM PE64 DLL), Linux (ELF64 SO), macOS (Mach-O dylib/NEON), Conda
# ---------------------------------------------------------------------------

def _bind_gemm_symbols(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Bind CTypes signatures for hardware-accelerated GEMM microkernels."""
    if hasattr(lib, "nanogemm_matmul"):
        lib.nanogemm_matmul.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.nanogemm_matmul.restype = None

    if hasattr(lib, "nanogemm_sgemm"):
        lib.nanogemm_sgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p, ctypes.c_int,
        ]
        lib.nanogemm_sgemm.restype = None

    if hasattr(lib, "nanogemm_simd_isa"):
        lib.nanogemm_simd_isa.argtypes = []
        lib.nanogemm_simd_isa.restype = ctypes.c_char_p

    if hasattr(lib, "nanogemm_bmm"):
        lib.nanogemm_bmm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        lib.nanogemm_bmm.restype = None

    if hasattr(lib, "nanogemm_gemm_i8i8i32"):
        lib.nanogemm_gemm_i8i8i32.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.nanogemm_gemm_i8i8i32.restype = None

    return lib


def _find_fasm_library() -> Optional[str]:
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    is_64bit = sys.maxsize > 2**32
    
    candidates = []
    if sys.platform.startswith("win"):
        dll_name = "nanogemm64.dll" if is_64bit else "nanogemm32.dll"
        candidates = [
            root_dir / "asm" / dll_name,
            pkg_dir / "asm" / dll_name,
            pkg_dir / dll_name,
            root_dir / "build" / dll_name,
            root_dir / dll_name,
        ]
    elif sys.platform.startswith("linux"):
        so_name = "nanogemm64.so" if is_64bit else "nanogemm32.so"
        candidates = [
            root_dir / "asm" / so_name,
            pkg_dir / "asm" / so_name,
            pkg_dir / so_name,
            root_dir / "build" / so_name,
            root_dir / so_name,
            Path("/usr/local/lib") / so_name,
            Path("/usr/lib") / so_name,
        ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None

_fasm_path = _find_fasm_library()
_fasm_lib = None
if _fasm_path:
    try:
        _fasm_lib = _bind_gemm_symbols(ctypes.CDLL(_fasm_path))
    except Exception:
        _fasm_lib = None


def _find_library() -> Optional[str]:
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    
    names = []
    if sys.platform.startswith("win"):
        names = ["nanogemm.dll", "nanogemm64.dll", "libnanogemm.dll"]
    elif sys.platform == "darwin":
        names = ["libnanogemm.dylib", "nanogemm.dylib"]
    else:
        names = ["libnanogemm.so", "nanogemm.so", "nanogemm64.so"]

    search_dirs = [
        pkg_dir,
        pkg_dir / "asm",
        root_dir / "asm",
        root_dir / "build",
        root_dir,
    ]

    # Conda environment prefix support
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cp = Path(conda_prefix)
        search_dirs.extend([
            cp / "lib",
            cp / "Library" / "bin",
            cp / "DLLs",
        ])

    # Virtualenv and standard Python prefix paths
    search_dirs.extend([
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix) / "bin",
    ])

    for d in search_dirs:
        for name in names:
            p = d / name
            if p.is_file():
                return str(p)

    # Dynamic library resolution via ctypes.util
    for name in ("nanogemm", "nanogemm64", "libnanogemm"):
        found = ctypes.util.find_library(name)
        if found:
            return found

    return None

_lib_path = _find_library()
_lib = None
if _lib_path:
    try:
        _lib = _bind_gemm_symbols(ctypes.CDLL(_lib_path))
    except Exception:
        _lib = None

# Unified native acceleration engine (FASM / C-Shared / Conda)
_native_lib = _fasm_lib or _lib

# ---------------------------------------------------------------------------
# Backend Management
# ---------------------------------------------------------------------------

_ACTIVE_BACKEND = "auto" if (_native_lib is not None or _HAS_C_EXT) else "auto"

def set_backend(backend: str) -> None:
    """
    Set the execution backend for NanoGEMM.

    Parameters
    ----------
    backend : str
        One of:
        - 'fasm' / 'native': Hardware-accelerated assembly/SIMD microkernel (AVX2/NEON)
        - 'c': Compiled C extension with SIMD intrinsics
        - 'auto': Automatically select optimal backend based on tensor dimensions
    """
    global _ACTIVE_BACKEND
    b = backend.lower().strip()
    if b not in ("fasm", "native", "c", "auto"):
        raise ValueError(f"Unknown backend '{backend}'. Choose from 'fasm', 'native', 'c', or 'auto'.")
    if b in ("fasm", "native") and _native_lib is None:
        raise RuntimeError("Native SIMD microkernel library not found.")
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
    if _native_lib is not None and hasattr(_native_lib, "nanogemm_simd_isa"):
        raw = _native_lib.nanogemm_simd_isa()
        return raw.decode("utf-8") if raw else "Native SIMD"
    return "Not compiled"


def matmul(
    a: Any,
    b: Any,
    out: Optional[Any] = None,
    backend: Optional[str] = None,
    shape: Optional[Tuple[int, int, int]] = None,
) -> Any:
    """
    Multiply two 2D matrices using hardware-accelerated SIMD GEMM.

    Supports:
    - NumPy ndarray (float32, 2D)
    - 2D nested lists: [[...], [...]]
    - Pre-allocated ctypes.Array buffers (zero-copy FastPath for HFT/trading)
    """
    # 0. Zero-Copy CTypes Array FastPath
    if isinstance(a, ctypes.Array) and isinstance(b, ctypes.Array) and shape is not None:
        M, N, K = shape
        if out is None:
            out = (ctypes.c_float * (M * N))()
        if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
            _native_lib.nanogemm_matmul(M, N, K, ctypes.byref(a), ctypes.byref(b), ctypes.byref(out))
            return out
        else:
            for i in range(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        s += float(a[i * K + k]) * float(b[k * N + j])
                    out[i * N + j] = s
            return out

    if np is None or not hasattr(a, "ndim") or not hasattr(b, "ndim"):
        if isinstance(a, ctypes.Array) and isinstance(b, ctypes.Array) and shape is not None:
            M, N, K = shape
            if out is None:
                out = (ctypes.c_float * (M * N))()
            if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
                _native_lib.nanogemm_matmul(M, N, K, ctypes.byref(a), ctypes.byref(b), ctypes.byref(out))
                return out
            else:
                for i in range(M):
                    for j in range(N):
                        s = 0.0
                        for k in range(K):
                            s += float(a[i * K + k]) * float(b[k * N + j])
                        out[i * N + j] = s
                return out

        if not isinstance(a, list) or not isinstance(b, list):
            raise TypeError("Expected numpy arrays, ctypes arrays, or 2D nested lists")
        M = len(a)
        K = len(a[0]) if M > 0 and isinstance(a[0], list) else 0
        K_b = len(b)
        N = len(b[0]) if K_b > 0 and isinstance(b[0], list) else 0
        if K != K_b or K == 0 or N == 0:
            raise ValueError(f"Incompatible matrix dimensions: cannot multiply ({M}, {K}) by ({K_b}, {N})")
        a_flat = [float(x) for row in a for x in row]
        b_flat = [float(x) for row in b for x in row]
        if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
            c_a = (ctypes.c_float * len(a_flat))(*a_flat)
            c_b = (ctypes.c_float * len(b_flat))(*b_flat)
            c_c = (ctypes.c_float * (M * N))()
            _native_lib.nanogemm_matmul(M, N, K, ctypes.byref(c_a), ctypes.byref(c_b), ctypes.byref(c_c))
            if out is not None and isinstance(out, ctypes.Array):
                ctypes.memmove(ctypes.byref(out), ctypes.byref(c_c), M * N * 4)
                return out
            res = []
            for i in range(M):
                res.append([c_c[i * N + j] for j in range(N)])
            return res
        else:
            res = []
            for i in range(M):
                row = []
                for j in range(N):
                    row.append(sum(a[i][k] * b[k][j] for k in range(K)))
                res.append(row)
            if out is not None and isinstance(out, ctypes.Array):
                for i in range(M):
                    for j in range(N):
                        out[i * N + j] = res[i][j]
                return out
            return res

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
        # for medium/large matrices (>= 64x64), native assembly kernel is up to 2x faster.
        if M <= 32 and N <= 32 and _HAS_C_EXT:
            chosen_backend = "c"
        elif _native_lib is not None:
            chosen_backend = "fasm" if _fasm_lib is not None else "native"
        elif _HAS_C_EXT:
            chosen_backend = "c"
        else:
            chosen_backend = "c"

    if chosen_backend in ("fasm", "native") and _native_lib is not None:
        _native_lib.nanogemm_matmul(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

    if chosen_backend == "c" and _HAS_C_EXT:
        _ext.matmul_fast(a, b, out)
        return out

    if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
        _native_lib.nanogemm_matmul(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

    if _HAS_C_EXT:
        _ext.matmul_fast(a, b, out)
        return out

    raise RuntimeError("NanoGEMM native binary not available. Please compile nanogemm native binary.")


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
    if np is None or not hasattr(a, "ndim") or not hasattr(b, "ndim"):
        if not isinstance(a, list) or not isinstance(b, list):
            raise TypeError("Expected numpy arrays or 2D nested lists")
        M = len(a)
        K = len(a[0]) if M > 0 and isinstance(a[0], list) else 0
        K_b = len(b)
        N = len(b[0]) if K_b > 0 and isinstance(b[0], list) else 0
        if K != K_b or K == 0 or N == 0:
            raise ValueError(f"Incompatible matrix dimensions: ({M}, {K}) vs ({K_b}, {N})")
        a_flat = [float(x) for row in a for x in row]
        b_flat = [float(x) for row in b for x in row]
        if _native_lib is not None and hasattr(_native_lib, "nanogemm_sgemm"):
            c_a = (ctypes.c_float * len(a_flat))(*a_flat)
            c_b = (ctypes.c_float * len(b_flat))(*b_flat)
            if c is not None and isinstance(c, list):
                c_flat = [float(x) for row in c for x in row]
                c_c = (ctypes.c_float * len(c_flat))(*c_flat)
            else:
                c_c = (ctypes.c_float * (M * N))()
            _native_lib.nanogemm_sgemm(
                M, N, K,
                float(alpha),
                ctypes.byref(c_a), K,
                ctypes.byref(c_b), N,
                float(beta),
                ctypes.byref(c_c), N
            )
            res = []
            for i in range(M):
                res.append([c_c[i * N + j] for j in range(N)])
            return res
        else:
            res = []
            for i in range(M):
                row = []
                for j in range(N):
                    dot = sum(a[i][k] * b[k][j] for k in range(K))
                    old_c = c[i][j] if (c and i < len(c) and j < len(c[i])) else 0.0
                    row.append(alpha * dot + beta * old_c)
                res.append(row)
            return res

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
        elif _native_lib is not None:
            chosen_backend = "fasm" if _fasm_lib is not None else "native"
        elif _HAS_C_EXT and hasattr(_ext, "sgemm_fast"):
            chosen_backend = "c"
        else:
            chosen_backend = "c"

    if chosen_backend in ("fasm", "native") and _native_lib is not None:
        _native_lib.nanogemm_sgemm(
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

    if _native_lib is not None and hasattr(_native_lib, "nanogemm_sgemm"):
        _native_lib.nanogemm_sgemm(
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
    backend: Optional[str] = None,
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
    if np is None or not hasattr(a, "shape") or not hasattr(b, "shape"):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) > 0 and isinstance(a[0], list) and len(a[0]) > 0 and isinstance(a[0][0], list):
                if len(b) > 0 and isinstance(b[0], list) and len(b[0]) > 0 and isinstance(b[0][0], list):
                    return [matmul(a[i], b[i], backend=backend) for i in range(len(a))]
                else:
                    return [matmul(a[i], b, backend=backend) for i in range(len(a))]
            elif len(b) > 0 and isinstance(b[0], list) and len(b[0]) > 0 and isinstance(b[0][0], list):
                return [matmul(a, b[i], backend=backend) for i in range(len(b))]
        raise TypeError("Expected numpy arrays or 3D nested lists for bmm")

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

    chosen_backend = backend.lower().strip() if backend else _ACTIVE_BACKEND

    if chosen_backend in ("fasm", "native") and _native_lib and hasattr(_native_lib, "nanogemm_bmm"):
        batch_count = out_shape_3d[0]
        M = a.shape[-2] if a.ndim >= 2 else 1
        K = a.shape[-1]
        N = b.shape[-1]
        stride_a = M * K if a.ndim == 3 else 0
        stride_b = K * N if b.ndim == 3 else 0
        stride_c = M * N
        _native_lib.nanogemm_bmm(
            batch_count, M, N, K,
            a.ctypes.data, b.ctypes.data, out_buf.ctypes.data,
            stride_a, stride_b, stride_c
        )
        if is_4d:
            return out_buf.reshape(out_shape_4d)
        return out_buf

    if _HAS_C_EXT and hasattr(_ext, "bmm_fast") and chosen_backend not in ("fasm", "native"):
        _ext.bmm_fast(a, b, out_buf)
    elif _native_lib and hasattr(_native_lib, "nanogemm_bmm"):
        batch_count = out_shape_3d[0]
        M = a.shape[-2] if a.ndim >= 2 else 1
        K = a.shape[-1]
        N = b.shape[-1]
        stride_a = M * K if a.ndim == 3 else 0
        stride_b = K * N if b.ndim == 3 else 0
        stride_c = M * N
        _native_lib.nanogemm_bmm(
            batch_count, M, N, K,
            a.ctypes.data, b.ctypes.data, out_buf.ctypes.data,
            stride_a, stride_b, stride_c
        )
    else:
        batch_count = out_shape_3d[0]
        for i in range(batch_count):
            a_slice = a[i] if a.ndim == 3 else a
            b_slice = b[i] if b.ndim == 3 else b
            out_buf[i] = matmul(a_slice, b_slice, backend=chosen_backend)

    if is_4d:
        return out_buf.reshape(out_shape_4d)
    return out_buf


def matmul_int8(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
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
    backend : str, optional
        Override execution backend ('fasm', 'c', or 'auto').

    Returns
    -------
    np.ndarray
        Result of matrix multiplication with int32 dtype and shape (M, N).
    """
    if np is None or not hasattr(a, "ndim") or not hasattr(b, "ndim"):
        if not isinstance(a, list) or not isinstance(b, list):
            raise TypeError("Expected numpy arrays or 2D nested lists")
        M = len(a)
        K = len(a[0]) if M > 0 and isinstance(a[0], list) else 0
        K_b = len(b)
        N = len(b[0]) if K_b > 0 and isinstance(b[0], list) else 0
        if K != K_b or K == 0 or N == 0:
            raise ValueError(f"Incompatible matrix dimensions: ({M}, {K}) vs ({K_b}, {N})")
        a_flat = [int(x) for row in a for x in row]
        b_flat = [int(x) for row in b for x in row]
        if _native_lib and hasattr(_native_lib, "nanogemm_gemm_i8i8i32"):
            c_a = (ctypes.c_int8 * len(a_flat))(*a_flat)
            c_b = (ctypes.c_int8 * len(b_flat))(*b_flat)
            c_c = (ctypes.c_int32 * (M * N))()
            _native_lib.nanogemm_gemm_i8i8i32(M, N, K, ctypes.byref(c_a), ctypes.byref(c_b), ctypes.byref(c_c))
            res = []
            for i in range(M):
                res.append([c_c[i * N + j] for j in range(N)])
            return res
        else:
            res = []
            for i in range(M):
                row = []
                for j in range(N):
                    row.append(sum(int(a[i][k]) * int(b[k][j]) for k in range(K)))
                res.append(row)
            return res

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

    chosen_backend = backend.lower().strip() if backend else _ACTIVE_BACKEND

    if chosen_backend in ("fasm", "native") and _native_lib and hasattr(_native_lib, "nanogemm_gemm_i8i8i32"):
        _native_lib.nanogemm_gemm_i8i8i32(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
        return out

    if _HAS_C_EXT and hasattr(_ext, "matmul_int8_fast") and chosen_backend not in ("fasm", "native"):
        _ext.matmul_int8_fast(a, b, out)
        return out

    if _native_lib and hasattr(_native_lib, "nanogemm_gemm_i8i8i32"):
        _native_lib.nanogemm_gemm_i8i8i32(M, N, K, a.ctypes.data, b.ctypes.data, out.ctypes.data)
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
        if out is None:
            res = res + bias
        else:
            np.add(res, bias, out=out)

    return res


def is_available() -> bool:
    """
    Check if hardware-accelerated SIMD (FASM assembly kernel or C extension) is operational.
    """
    return (_native_lib is not None) or _HAS_C_EXT


def matmul_flat(
    a_flat: Any,
    b_flat: Any,
    M: int,
    N: int,
    K: int,
    out: Optional[Any] = None,
) -> Any:
    """
    Fast direct matrix multiplication on 1D contiguous sequences / ctypes arrays.
    Multiplies (M x K) by (K x N) -> (M x N).
    """
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: M={M}, N={N}, K={K}")

    if isinstance(a_flat, ctypes.Array):
        c_a = a_flat
    elif isinstance(a_flat, (list, tuple)):
        c_a = (ctypes.c_float * len(a_flat))(*a_flat)
    elif np is not None and isinstance(a_flat, np.ndarray):
        c_a = a_flat.ctypes.data_as(ctypes.c_void_p)
    else:
        c_a = (ctypes.c_float * len(a_flat))(*a_flat)

    if isinstance(b_flat, ctypes.Array):
        c_b = b_flat
    elif isinstance(b_flat, (list, tuple)):
        c_b = (ctypes.c_float * len(b_flat))(*b_flat)
    elif np is not None and isinstance(b_flat, np.ndarray):
        c_b = b_flat.ctypes.data_as(ctypes.c_void_p)
    else:
        c_b = (ctypes.c_float * len(b_flat))(*b_flat)

    total_out = M * N
    if out is None:
        c_out = (ctypes.c_float * total_out)()
    elif isinstance(out, ctypes.Array):
        c_out = out
    elif np is not None and isinstance(out, np.ndarray):
        c_out = out.ctypes.data_as(ctypes.c_void_p)
    else:
        c_out = (ctypes.c_float * total_out)()

    if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
        ptr_a = c_a if isinstance(c_a, ctypes.c_void_p) else ctypes.byref(c_a)
        ptr_b = c_b if isinstance(c_b, ctypes.c_void_p) else ctypes.byref(c_b)
        ptr_out = c_out if isinstance(c_out, ctypes.c_void_p) else ctypes.byref(c_out)
        _native_lib.nanogemm_matmul(M, N, K, ptr_a, ptr_b, ptr_out)
    else:
        for i in range(M):
            for j in range(N):
                s = sum(a_flat[i * K + k] * b_flat[k * N + j] for k in range(K))
                c_out[i * N + j] = s

    if out is not None and out is c_out:
        return out
    if np is not None and isinstance(out, np.ndarray):
        return out
    return c_out


def score_batch(
    features: Any,
    weights: Any,
    bias: float = 0.0,
    activation: Optional[str] = "sigmoid",
    out: Optional[Any] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Any:
    """
    Sub-microsecond batched AI scoring with AVX2 SIMD NanoGEMM kernel.
    Computes: Z = features @ weights + bias, followed by activation (sigmoid/linear/relu).

    Parameters
    ----------
    features : np.ndarray | list[list[float]] | list[float] | ctypes.Array
        Feature matrix of shape (N, K), or flat buffer with shape=(N, K).
    weights : np.ndarray | list[float] | list[list[float]] | ctypes.Array
        Model weights of shape (K,) or (K, 1).
    bias : float, default 0.0
        Model bias term.
    activation : str, default "sigmoid"
        Activation function: "sigmoid", "linear" (or None), or "relu".
    out : optional buffer
        Pre-allocated output buffer of length N.
    shape : tuple of (N, K), optional
        Required if features is a pre-flattened 1D sequence or ctypes array.

    Returns
    -------
    np.ndarray (if input was numpy) or list[float] (if input was list/ctypes)
        Confidence scores of length N.
    """
    # 1. NumPy FastPath
    if np is not None and isinstance(features, np.ndarray):
        if features.ndim != 2:
            raise ValueError(f"features must be 2D array, got {features.ndim}D")
        N, K = features.shape

        if isinstance(weights, np.ndarray):
            w = weights.reshape((K, 1)) if weights.ndim == 1 else weights
        else:
            w = np.array(weights, dtype=np.float32).reshape((K, 1))

        if out is None or not isinstance(out, np.ndarray):
            out_buf = np.empty((N, 1), dtype=np.float32, order="C")
        else:
            out_buf = out.reshape((N, 1)) if out.ndim == 1 else out

        matmul(features, w, out=out_buf)

        if bias != 0.0:
            out_buf += np.float32(bias)

        if activation == "sigmoid":
            np.clip(out_buf, -15.0, 15.0, out=out_buf)
            res = 1.0 / (1.0 + np.exp(-out_buf))
            return res.ravel()
        elif activation == "relu":
            np.maximum(out_buf, 0.0, out=out_buf)
            return out_buf.ravel()
        else:
            return out_buf.ravel()

    # 2. Pure Python / List / CTypes FastPath (No-NumPy / PyPy / Standalone)
    import math

    if shape is not None:
        N, K = shape
        if isinstance(features, (list, tuple)):
            flat_x = [float(v) for v in features]
            c_x = (ctypes.c_float * len(flat_x))(*flat_x)
        else:
            c_x = features
    elif isinstance(features, (list, tuple)):
        N = len(features)
        if N == 0:
            return []
        if isinstance(features[0], (list, tuple)):
            K = len(features[0])
            flat_x = [float(v) for row in features for v in row]
            c_x = (ctypes.c_float * len(flat_x))(*flat_x)
        else:
            raise ValueError("shape=(N, K) required when features is a 1D flat sequence")
    elif isinstance(features, ctypes.Array):
        if shape is None:
            raise ValueError("shape=(N, K) required when features is a ctypes.Array")
        N, K = shape
        c_x = features
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")

    if isinstance(weights, (list, tuple)):
        if len(weights) > 0 and isinstance(weights[0], (list, tuple)):
            flat_w = [float(w[0]) for w in weights]
        else:
            flat_w = [float(w) for w in weights]
        if len(flat_w) != K:
            raise ValueError(f"Weight length {len(flat_w)} does not match feature dimension {K}")
        c_w = (ctypes.c_float * K)(*flat_w)
    elif isinstance(weights, ctypes.Array):
        flat_w = [float(weights[i]) for i in range(K)]
        c_w = weights
    else:
        raise TypeError(f"Unsupported weights type: {type(weights)}")

    if out is not None and isinstance(out, ctypes.Array) and len(out) >= N:
        c_out = out
    else:
        c_out = (ctypes.c_float * N)()

    if _native_lib is not None and hasattr(_native_lib, "nanogemm_matmul"):
        _native_lib.nanogemm_matmul(N, 1, K, ctypes.byref(c_x), ctypes.byref(c_w), ctypes.byref(c_out))
    else:
        for i in range(N):
            c_out[i] = sum(c_x[i * K + k] * flat_w[k] for k in range(K))

    b_val = float(bias)
    if activation == "sigmoid":
        res = [1.0 / (1.0 + math.exp(-max(min(c_out[i] + b_val, 15.0), -15.0))) for i in range(N)]
    elif activation == "relu":
        res = [max(0.0, c_out[i] + b_val) for i in range(N)]
    else:
        res = [c_out[i] + b_val for i in range(N)]

    return res

