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

static PyObject* py_nanogemm_isa(PyObject* self, PyObject* Py_UNUSED(args)) {
    return PyUnicode_FromString(nanogemm_simd_isa());
}

static PyMethodDef NanoGemmMethods[] = {
    {"matmul_fast", py_nanogemm_matmul, METH_VARARGS, "Fast SIMD GEMM via Python buffer protocol"},
    {"sgemm_fast", py_nanogemm_sgemm, METH_VARARGS, "Fast BLAS SGEMM via Python buffer protocol"},
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
