#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "nanogemm_kernel.h"

static PyObject* py_nanogemm_matmul(PyObject* self, PyObject* args) {
    PyObject *obj_a, *obj_b, *obj_out = Py_None;

    if (!PyArg_ParseTuple(args, "OO|O", &obj_a, &obj_b, &obj_out)) {
        return NULL;
    }

    Py_buffer buf_a, buf_b, buf_out;
    if (PyObject_GetBuffer(obj_a, &buf_a, PyBUF_ND | PyBUF_STRIDES) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(obj_b, &buf_b, PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        return NULL;
    }

    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyErr_SetString(PyExc_ValueError, "Expected 2D arrays");
        return NULL;
    }

    int M = (int)buf_a.shape[0];
    int K = (int)buf_a.shape[1];
    int K_b = (int)buf_b.shape[0];
    int N = (int)buf_b.shape[1];

    if (K != K_b) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: (%d,%d) x (%d,%d)", M, K, K_b, N);
        return NULL;
    }

    if (obj_out != Py_None) {
        if (PyObject_GetBuffer(obj_out, &buf_out, PyBUF_WRITABLE | PyBUF_ND | PyBUF_STRIDES) != 0) {
            PyBuffer_Release(&buf_a);
            PyBuffer_Release(&buf_b);
            return NULL;
        }

        /* Release GIL during heavy SIMD matrix computation */
        Py_BEGIN_ALLOW_THREADS
        nanogemm_matmul(M, N, K, (const float*)buf_a.buf, (const float*)buf_b.buf, (float*)buf_out.buf);
        Py_END_ALLOW_THREADS

        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);

        Py_INCREF(obj_out);
        return obj_out;
    }

    PyBuffer_Release(&buf_a);
    PyBuffer_Release(&buf_b);
    PyErr_SetString(PyExc_NotImplementedError, "Pass pre-allocated out buffer for maximum speed");
    return NULL;
}

static PyObject* py_nanogemm_sgemm(PyObject* self, PyObject* args) {
    PyObject *obj_a, *obj_b, *obj_c;
    float alpha = 1.0f, beta = 0.0f;

    if (!PyArg_ParseTuple(args, "OOffO", &obj_a, &obj_b, &alpha, &beta, &obj_c)) {
        return NULL;
    }

    Py_buffer buf_a, buf_b, buf_c;
    if (PyObject_GetBuffer(obj_a, &buf_a, PyBUF_ND | PyBUF_STRIDES) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(obj_b, &buf_b, PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        return NULL;
    }
    if (PyObject_GetBuffer(obj_c, &buf_c, PyBUF_WRITABLE | PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        return NULL;
    }

    if (buf_a.ndim != 2 || buf_b.ndim != 2 || buf_c.ndim != 2) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_c);
        PyErr_SetString(PyExc_ValueError, "Expected 2D arrays");
        return NULL;
    }

    int M = (int)buf_a.shape[0];
    int K = (int)buf_a.shape[1];
    int K_b = (int)buf_b.shape[0];
    int N = (int)buf_b.shape[1];

    if (K != K_b) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_c);
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: (%d,%d) x (%d,%d)", M, K, K_b, N);
        return NULL;
    }

    /* Release GIL during heavy SIMD BLAS SGEMM computation */
    Py_BEGIN_ALLOW_THREADS
    nanogemm_sgemm(M, N, K, alpha, (const float*)buf_a.buf, K, (const float*)buf_b.buf, N, beta, (float*)buf_c.buf, N);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf_a);
    PyBuffer_Release(&buf_b);
    PyBuffer_Release(&buf_c);

    Py_INCREF(obj_c);
    return obj_c;
}

static PyObject* py_nanogemm_bmm(PyObject* self, PyObject* args) {
    PyObject *obj_a, *obj_b, *obj_out;

    if (!PyArg_ParseTuple(args, "OOO", &obj_a, &obj_b, &obj_out)) {
        return NULL;
    }

    Py_buffer buf_a, buf_b, buf_out;
    if (PyObject_GetBuffer(obj_a, &buf_a, PyBUF_ND | PyBUF_STRIDES) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(obj_b, &buf_b, PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        return NULL;
    }
    if (PyObject_GetBuffer(obj_out, &buf_out, PyBUF_WRITABLE | PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        return NULL;
    }

    int batch_count = 0;
    int M = 0, K = 0, N = 0;
    int stride_a = 0, stride_b = 0, stride_c = 0;

    if (buf_a.ndim == 3 && buf_b.ndim == 3) {
        if (buf_a.shape[0] != buf_b.shape[0]) {
            PyBuffer_Release(&buf_a);
            PyBuffer_Release(&buf_b);
            PyBuffer_Release(&buf_out);
            PyErr_SetString(PyExc_ValueError, "Batch dimension mismatch in bmm");
            return NULL;
        }
        batch_count = (int)buf_a.shape[0];
        M = (int)buf_a.shape[1];
        K = (int)buf_a.shape[2];
        if (K != (int)buf_b.shape[1]) {
            PyBuffer_Release(&buf_a);
            PyBuffer_Release(&buf_b);
            PyBuffer_Release(&buf_out);
            PyErr_SetString(PyExc_ValueError, "Inner dimension K mismatch in bmm");
            return NULL;
        }
        N = (int)buf_b.shape[2];
        stride_a = M * K;
        stride_b = K * N;
    } else if (buf_a.ndim == 2 && buf_b.ndim == 3) {
        batch_count = (int)buf_b.shape[0];
        M = (int)buf_a.shape[0];
        K = (int)buf_a.shape[1];
        if (K != (int)buf_b.shape[1]) {
            PyBuffer_Release(&buf_a);
            PyBuffer_Release(&buf_b);
            PyBuffer_Release(&buf_out);
            PyErr_SetString(PyExc_ValueError, "Inner dimension K mismatch in bmm");
            return NULL;
        }
        N = (int)buf_b.shape[2];
        stride_a = 0;
        stride_b = K * N;
    } else if (buf_a.ndim == 3 && buf_b.ndim == 2) {
        batch_count = (int)buf_a.shape[0];
        M = (int)buf_a.shape[1];
        K = (int)buf_a.shape[2];
        if (K != (int)buf_b.shape[0]) {
            PyBuffer_Release(&buf_a);
            PyBuffer_Release(&buf_b);
            PyBuffer_Release(&buf_out);
            PyErr_SetString(PyExc_ValueError, "Inner dimension K mismatch in bmm");
            return NULL;
        }
        N = (int)buf_b.shape[1];
        stride_a = M * K;
        stride_b = 0;
    } else {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_SetString(PyExc_ValueError, "bmm expects 3D tensors (or 2D broadcast against 3D)");
        return NULL;
    }

    if (buf_out.ndim != 3 || buf_out.shape[0] != batch_count || buf_out.shape[1] != M || buf_out.shape[2] != N) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_Format(PyExc_ValueError, "Output buffer shape must be (%d, %d, %d)", batch_count, M, N);
        return NULL;
    }
    stride_c = M * N;

    Py_BEGIN_ALLOW_THREADS
    nanogemm_bmm(batch_count, M, N, K,
                 (const float*)buf_a.buf, stride_a,
                 (const float*)buf_b.buf, stride_b,
                 (float*)buf_out.buf, stride_c);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf_a);
    PyBuffer_Release(&buf_b);
    PyBuffer_Release(&buf_out);

    Py_INCREF(obj_out);
    return obj_out;
}

static PyObject* py_nanogemm_matmul_int8(PyObject* self, PyObject* args) {
    PyObject *obj_a, *obj_b, *obj_out;

    if (!PyArg_ParseTuple(args, "OOO", &obj_a, &obj_b, &obj_out)) {
        return NULL;
    }

    Py_buffer buf_a, buf_b, buf_out;
    if (PyObject_GetBuffer(obj_a, &buf_a, PyBUF_ND | PyBUF_STRIDES) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(obj_b, &buf_b, PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        return NULL;
    }
    if (PyObject_GetBuffer(obj_out, &buf_out, PyBUF_WRITABLE | PyBUF_ND | PyBUF_STRIDES) != 0) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        return NULL;
    }

    if (buf_a.ndim != 2 || buf_b.ndim != 2 || buf_out.ndim != 2) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_SetString(PyExc_ValueError, "Expected 2D arrays for matmul_int8");
        return NULL;
    }

    if (buf_a.itemsize != 1 || buf_b.itemsize != 1 || buf_out.itemsize != 4) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_SetString(PyExc_TypeError, "matmul_int8 requires int8 inputs (itemsize 1) and int32 output (itemsize 4)");
        return NULL;
    }

    int M = (int)buf_a.shape[0];
    int K = (int)buf_a.shape[1];
    int K_b = (int)buf_b.shape[0];
    int N = (int)buf_b.shape[1];

    if (K != K_b) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_Format(PyExc_ValueError, "Dimension mismatch in matmul_int8: (%d,%d) x (%d,%d)", M, K, K_b, N);
        return NULL;
    }

    if ((int)buf_out.shape[0] != M || (int)buf_out.shape[1] != N) {
        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);
        PyErr_Format(PyExc_ValueError, "Output buffer shape mismatch: expected (%d, %d)", M, N);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    nanogemm_gemm_i8i8i32(M, N, K,
                          (const int8_t*)buf_a.buf, K,
                          (const int8_t*)buf_b.buf, N,
                          (int32_t*)buf_out.buf, N);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf_a);
    PyBuffer_Release(&buf_b);
    PyBuffer_Release(&buf_out);

    Py_INCREF(obj_out);
    return obj_out;
}

static PyObject* py_nanogemm_isa(PyObject* self, PyObject* Py_UNUSED(args)) {
    return PyUnicode_FromString(nanogemm_simd_isa());
}

static PyMethodDef NanoGemmMethods[] = {
    {"matmul_fast", py_nanogemm_matmul, METH_VARARGS, "Fast SIMD GEMM via Python buffer protocol"},
    {"sgemm_fast", py_nanogemm_sgemm, METH_VARARGS, "Fast BLAS SGEMM via Python buffer protocol"},
    {"bmm_fast", py_nanogemm_bmm, METH_VARARGS, "Fast Batched SIMD GEMM via Python buffer protocol"},
    {"matmul_int8_fast", py_nanogemm_matmul_int8, METH_VARARGS, "Fast Quantized INT8 SIMD GEMM"},
    {"simd_isa", py_nanogemm_isa, METH_NOARGS, "Active SIMD instruction set"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nanogemm_ext_module = {
    PyModuleDef_HEAD_INIT,
    "_ext",
    "NanoGEMM C extension module",
    -1,
    NanoGemmMethods
};

PyMODINIT_FUNC PyInit__ext(void) {
    PyObject* m = PyModule_Create(&nanogemm_ext_module);
    if (m == NULL) {
        return NULL;
    }
#if defined(Py_GIL_DISABLED) && defined(Py_MOD_GIL_NOT_USED)
    /* PEP 703 Free-Threaded (No-GIL) CPython 3.13 / 3.14 / 3.15 declaration */
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
    return m;
}
