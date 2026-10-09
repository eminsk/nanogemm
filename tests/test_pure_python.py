"""
Tests for nanogemm running in pure Python / PyPy environments (without NumPy dependency).
Verifies fallback & native CTypes list execution, dimension validation, and fastpaths.
"""

import pytest
import ctypes
import nanogemm
from nanogemm import matmul, sgemm, bmm, matmul_int8


def test_matmul_nested_lists():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19.0, 22.0], [43.0, 50.0]]
    res = matmul(a, b)
    assert len(res) == 2
    assert len(res[0]) == 2
    assert pytest.approx(res[0][0]) == 19.0
    assert pytest.approx(res[0][1]) == 22.0
    assert pytest.approx(res[1][0]) == 43.0
    assert pytest.approx(res[1][1]) == 50.0


def test_matmul_ctypes_array():
    M, K, N = 2, 2, 2
    a_arr = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    b_arr = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
    out_arr = (ctypes.c_float * 4)()

    res = matmul(a_arr, b_arr, out=out_arr, shape=(M, N, K))
    assert res is out_arr
    assert pytest.approx(res[0]) == 19.0
    assert pytest.approx(res[1]) == 22.0
    assert pytest.approx(res[2]) == 43.0
    assert pytest.approx(res[3]) == 50.0


def test_sgemm_nested_lists():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    c = [[1.0, 1.0], [1.0, 1.0]]
    # 2.0 * (A @ B) + 0.5 * C = 2.0 * [[19, 22], [43, 50]] + [[0.5, 0.5], [0.5, 0.5]]
    # = [[38.5, 44.5], [86.5, 100.5]]
    res = sgemm(a, b, c=c, alpha=2.0, beta=0.5)
    assert pytest.approx(res[0][0]) == 38.5
    assert pytest.approx(res[0][1]) == 44.5
    assert pytest.approx(res[1][0]) == 86.5
    assert pytest.approx(res[1][1]) == 100.5


def test_bmm_nested_lists():
    a = [[[1.0, 2.0], [3.0, 4.0]]]
    b = [[[5.0, 6.0], [7.0, 8.0]]]
    res = bmm(a, b)
    assert len(res) == 1
    assert pytest.approx(res[0][0][0]) == 19.0
    assert pytest.approx(res[0][1][1]) == 50.0


def test_matmul_int8_nested_lists():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    res = matmul_int8(a, b)
    assert res == [[19, 22], [43, 50]]


def test_dimension_mismatch():
    a = [[1.0, 2.0, 3.0]]
    b = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError):
        matmul(a, b)
