"""
Native Model Context Protocol (MCP) Server for NanoGEMM.
Exposes bare-metal SIMD & Assembly matrix multiplication (GEMM),
CPU hardware micro-benchmarking, and SIMD ISA inspection
to Claude Desktop, Cursor, Windsurf, Antigravity, and any MCP client over JSON-RPC 2.0 stdio.

Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import nanogemm
from nanogemm import __version__

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "gemm_benchmark",
        "title": "Benchmark NanoGEMM vs NumPy CPU Matrix Multiplication",
        "description": (
            "Benchmark CPU matrix multiplication (M x K x N) comparing NanoGEMM's bare-metal "
            "SIMD (AVX2+FMA / ARM NEON / FASM) microkernel against standard NumPy BLAS. "
            "Reports execution latency in microseconds, GFLOPS, and speedup ratio."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "default": 128, "description": "Number of rows of matrix A and C (default: 128)"},
                "k": {"type": "integer", "default": 128, "description": "Number of columns of A and rows of B (default: 128)"},
                "n": {"type": "integer", "default": 128, "description": "Number of columns of B and C (default: 128)"},
                "iterations": {"type": "integer", "default": 20, "description": "Number of timing benchmark iterations (default: 20)"},
            },
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer"},
                "k": {"type": "integer"},
                "n": {"type": "integer"},
                "iterations": {"type": "integer"},
                "nanogemm_us": {"type": "number"},
                "numpy_us": {"type": "number"},
                "speedup_ratio": {"type": "string"},
                "nanogemm_gflops": {"type": "number"},
                "simd_isa": {"type": "string"},
                "backend": {"type": "string"},
            },
        },
    },
    {
        "name": "gemm_multiply",
        "title": "Multiply Two Matrices with Bare-Metal SIMD",
        "description": (
            "Multiply two floating-point matrices (C = alpha * (A @ B) + beta * C) using NanoGEMM's "
            "bare-metal C/AVX2/NEON SIMD execution engine."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "matrix_a": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "2D numeric matrix A (M x K)",
                },
                "matrix_b": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "2D numeric matrix B (K x N)",
                },
                "alpha": {"type": "number", "default": 1.0, "description": "Scalar multiplier for A @ B (default: 1.0)"},
                "beta": {"type": "number", "default": 0.0, "description": "Scalar multiplier for C accumulator (default: 0.0)"},
            },
            "required": ["matrix_a", "matrix_b"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer"}},
                "matrix_c": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "frobenius_norm": {"type": "number"},
                "trace": {"type": "number"},
            },
        },
    },
    {
        "name": "gemm_hardware_info",
        "title": "Get CPU SIMD ISA & Microkernel Architecture Info",
        "description": (
            "Retrieve detected CPU SIMD instruction set architecture (AVX2, FMA, ARM NEON, SSE2, Scalar), "
            "active NanoGEMM execution backend, Python Free-Threaded (PEP 703 No-GIL) mode, and footprint."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "simd_isa": {"type": "string"},
                "active_backend": {"type": "string"},
                "is_free_threaded": {"type": "boolean"},
                "python_version": {"type": "string"},
                "platform": {"type": "string"},
                "footprint_kb": {"type": "number"},
            },
        },
    },
]


class NanoGEMMMCPServer:
    """Model Context Protocol (MCP) JSON-RPC 2.0 stdio server for NanoGEMM."""

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 message and return a response dict (or None for notifications)."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if req_id is None and (method.startswith("notifications/") or method == "initialized"):
            return None

        try:
            if method == "initialize":
                requested_version = params.get("protocolVersion")
                negotiated_version = (
                    requested_version
                    if isinstance(requested_version, str) and requested_version in SUPPORTED_PROTOCOL_VERSIONS
                    else LATEST_PROTOCOL_VERSION
                )
                result = {
                    "protocolVersion": negotiated_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "nanogemm-mcp",
                        "version": __version__,
                    },
                    "instructions": (
                        "NanoGEMM is a minimalist, bare-metal SIMD & Assembly General Matrix Multiply engine. "
                        "Use 'gemm_benchmark' to measure CPU matrix multiply speed vs NumPy, 'gemm_multiply' "
                        "for fast zero-JIT AVX2 matrix operations, and 'gemm_hardware_info' to detect SIMD ISA."
                    ),
                }
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

            if method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments") or {}
                with contextlib.redirect_stdout(sys.stderr):
                    tool_output = self._call_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(tool_output, ensure_ascii=False, indent=2),
                            }
                        ],
                        "isError": False,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "gemm_benchmark":
            m = max(16, min(1024, int(args.get("m", 128))))
            k = max(16, min(1024, int(args.get("k", 128))))
            n = max(16, min(1024, int(args.get("n", 128))))
            iters = max(1, min(100, int(args.get("iterations", 20))))

            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)

            # Warmup
            nanogemm.matmul(a, b)
            np.dot(a, b)

            # NanoGEMM timing
            t0 = time.perf_counter()
            for _ in range(iters):
                c_nano = nanogemm.matmul(a, b)
            t_nano_us = ((time.perf_counter() - t0) / iters) * 1e6

            # NumPy timing
            t0 = time.perf_counter()
            for _ in range(iters):
                c_np = np.dot(a, b)
            t_np_us = ((time.perf_counter() - t0) / iters) * 1e6

            ops = 2.0 * m * k * n
            gflops = (ops / (t_nano_us * 1e-6)) / 1e9 if t_nano_us > 0 else 0.0
            ratio = t_np_us / t_nano_us if t_nano_us > 0 else 1.0

            return {
                "m": m,
                "k": k,
                "n": n,
                "iterations": iters,
                "nanogemm_us": round(t_nano_us, 2),
                "numpy_us": round(t_np_us, 2),
                "speedup_ratio": f"{ratio:.2f}x",
                "nanogemm_gflops": round(gflops, 2),
                "simd_isa": nanogemm.get_simd_isa(),
                "backend": nanogemm.get_backend(),
            }

        if name == "gemm_multiply":
            raw_a = args.get("matrix_a")
            raw_b = args.get("matrix_b")
            if raw_a is None or raw_b is None:
                raise ValueError("Arguments 'matrix_a' and 'matrix_b' are required.")

            a = np.array(raw_a, dtype=np.float32)
            b = np.array(raw_b, dtype=np.float32)
            alpha = float(args.get("alpha", 1.0))
            beta = float(args.get("beta", 0.0))

            if a.ndim != 2 or b.ndim != 2:
                raise ValueError(f"Matrices must be 2D, got A: {a.shape}, B: {b.shape}")
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"Matrix dimension mismatch: A is {a.shape}, B is {b.shape} (inner dimensions must match)")

            if beta == 0.0 and alpha == 1.0:
                c = nanogemm.matmul(a, b)
            else:
                c = nanogemm.sgemm(a, b, alpha=alpha, beta=beta)

            frobenius_norm = float(np.linalg.norm(c))
            trace_val = float(np.trace(c)) if c.shape[0] == c.shape[1] else None

            # For small matrices return full matrix; for large ones return truncated preview
            if c.shape[0] <= 32 and c.shape[1] <= 32:
                res_matrix = c.tolist()
            else:
                res_matrix = c[:8, :8].tolist()

            return {
                "shape": list(c.shape),
                "matrix_c": res_matrix,
                "frobenius_norm": round(frobenius_norm, 4),
                "trace": round(trace_val, 4) if trace_val is not None else None,
            }

        if name == "gemm_hardware_info":
            is_nogil = getattr(sys, "_is_gil_enabled", lambda: True)() is False
            return {
                "version": __version__,
                "simd_isa": nanogemm.get_simd_isa(),
                "active_backend": nanogemm.get_backend(),
                "is_free_threaded": is_nogil,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "footprint_kb": 105.0,
            }

        raise ValueError(f"Unknown tool name: {name}")

    def run_stdio(self) -> None:
        """Run the JSON-RPC stdio event loop."""
        if sys.platform == "win32":
            try:
                import msvcrt
                msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
                msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
            except Exception:
                pass

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                line_stripped = line.strip()
                if not line_stripped:
                    continue

                try:
                    request = json.loads(line_stripped)
                except json.JSONDecodeError:
                    continue

                response = self.handle_request(request)
                if response is not None:
                    out = json.dumps(response, ensure_ascii=False) + "\n"
                    sys.stdout.write(out)
                    sys.stdout.flush()

            except (KeyboardInterrupt, BrokenPipeError):
                break
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main_mcp() -> None:
    """Entry point for the nanogemm-mcp command."""
    server = NanoGEMMMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main_mcp()
