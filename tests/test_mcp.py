"""
Unit Tests for NanoGEMM Native MCP Server (JSON-RPC 2.0 stdio)
"""

import json
import pytest

from nanogemm import __version__
from nanogemm.mcp_server import (
    NanoGEMMMCPServer,
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)


def test_mcp_initialize_negotiation():
    server = NanoGEMMMCPServer()

    # 1. Request latest supported version
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25"},
    }
    resp = server.handle_request(req)
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-11-25"
    assert resp["result"]["serverInfo"]["name"] == "nanogemm-mcp"
    assert resp["result"]["serverInfo"]["version"] == __version__
    assert "tools" in resp["result"]["capabilities"]

    # 2. Request older supported version
    req_old = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }
    resp_old = server.handle_request(req_old)
    assert resp_old["result"]["protocolVersion"] == "2024-11-05"

    # 3. Request unknown version -> fallback to latest supported
    req_unknown = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "initialize",
        "params": {"protocolVersion": "9999-99-99"},
    }
    resp_unknown = server.handle_request(req_unknown)
    assert resp_unknown["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


def test_mcp_tools_list_schema():
    server = NanoGEMMMCPServer()
    resp = server.handle_request({"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
    assert resp["id"] == 10
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "gemm_benchmark" in tool_names
    assert "gemm_multiply" in tool_names
    assert "gemm_hardware_info" in tool_names

    for tool in tools:
        assert "name" in tool
        assert "title" in tool
        assert "description" in tool
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]
        assert "inputSchema" in tool
        assert "outputSchema" in tool


def test_mcp_gemm_benchmark():
    server = NanoGEMMMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "gemm_benchmark",
            "arguments": {
                "m": 64,
                "k": 64,
                "n": 64,
                "iterations": 10,
            },
        },
    }
    resp = server.handle_request(req)
    assert resp["id"] == 20
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["m"] == 64
    assert content["k"] == 64
    assert content["n"] == 64
    assert content["nanogemm_us"] > 0.0
    assert content["numpy_us"] > 0.0
    assert "simd_isa" in content


def test_mcp_gemm_multiply():
    server = NanoGEMMMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 30,
        "method": "tools/call",
        "params": {
            "name": "gemm_multiply",
            "arguments": {
                "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
                "matrix_b": [[2.0, 0.0], [1.0, 2.0]],
            },
        },
    }
    resp = server.handle_request(req)
    assert resp["id"] == 30
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["shape"] == [2, 2]
    # [[1*2 + 2*1, 1*0 + 2*2], [3*2 + 4*1, 3*0 + 4*2]] = [[4, 4], [10, 8]]
    expected = [[4.0, 4.0], [10.0, 8.0]]
    for r in range(2):
        for c in range(2):
            assert abs(content["matrix_c"][r][c] - expected[r][c]) < 1e-4


def test_mcp_gemm_hardware_info():
    server = NanoGEMMMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 40,
        "method": "tools/call",
        "params": {"name": "gemm_hardware_info", "arguments": {}},
    }
    resp = server.handle_request(req)
    assert resp["id"] == 40
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["version"] == __version__
    assert "simd_isa" in content
    assert "active_backend" in content
