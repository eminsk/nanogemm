"""
NanoGEMM Command Line Interface (CLI)
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import argparse
import sys

from nanogemm import __version__
from nanogemm.mcp_server import main_mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nanogemm",
        description="Minimalist, bare-metal SIMD & Assembly GEMM engine for Python",
    )
    parser.add_argument("--version", "-v", action="version", version=f"nanogemm {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # mcp command
    subparsers.add_parser("mcp", help="Run NanoGEMM Model Context Protocol (MCP) server over stdio")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run CPU matrix multiplication benchmark vs NumPy")
    p_bench.add_argument("--size", "-s", type=int, default=256, help="Matrix dimension M=K=N (default: 256)")
    p_bench.add_argument("--iterations", "-i", type=int, default=30, help="Benchmark iterations (default: 30)")

    # info command
    subparsers.add_parser("info", help="Print detected SIMD architecture and compiler info")

    args = parser.parse_args()
    if args.command == "mcp":
        main_mcp()
    elif args.command == "benchmark":
        from nanogemm.mcp_server import NanoGEMMMCPServer
        srv = NanoGEMMMCPServer()
        res = srv._call_tool("gemm_benchmark", {"m": args.size, "k": args.size, "n": args.size, "iterations": args.iterations})
        print(f"\n⚡ NanoGEMM CPU Matrix Benchmark ({args.size}x{args.size}x{args.size}, {args.iterations} iters):")
        print(f"  • NanoGEMM:      {res['nanogemm_us']:.2f} µs ({res['nanogemm_gflops']:.2f} GFLOPS)")
        print(f"  • NumPy:         {res['numpy_us']:.2f} µs")
        print(f"  • Speedup Ratio: {res['speedup_ratio']}")
        print(f"  • SIMD ISA:      {res['simd_isa']}")
        print(f"  • Active Engine: {res['backend']}\n")
    elif args.command == "info":
        from nanogemm.mcp_server import NanoGEMMMCPServer
        srv = NanoGEMMMCPServer()
        info = srv._call_tool("gemm_hardware_info", {})
        print(f"\n⚡ NanoGEMM Hardware Architecture Info:")
        print(f"  • Version:       {info['version']}")
        print(f"  • SIMD ISA:      {info['simd_isa']}")
        print(f"  • Backend:       {info['active_backend']}")
        print(f"  • No-GIL (Free): {info['is_free_threaded']}")
        print(f"  • Python:        {info['python_version']}")
        print(f"  • Platform:      {info['platform']}")
        print(f"  • Footprint:     ~{info['footprint_kb']} KB\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
