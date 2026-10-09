"""
Pytest configuration for NanoGEMM test suite.
Safely ignores tests requiring NumPy or Windows FASM binaries when dependencies are absent.
"""

import sys

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    path_str = str(collection_path).lower()

    # Ignore Windows-specific FASM tests on non-Windows platforms
    if "test_fasm" in path_str and sys.platform != "win32":
        return True

    # If NumPy is not installed, ignore test files that depend on NumPy
    try:
        import numpy  # noqa: F401
    except ImportError:
        if any(name in path_str for name in ("test_bmm_int8", "test_correctness", "test_fasm")):
            return True

    return False
