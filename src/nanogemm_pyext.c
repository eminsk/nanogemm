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

        nanogemm_matmul(M, N, K, (const float*)buf_a.buf, (const float*)buf_b.buf, (float*)buf_out.buf);

        PyBuffer_Release(&buf_a);
        PyBuffer_Release(&buf_b);
        PyBuffer_Release(&buf_out);

        Py_INCREF(obj_out);
        return obj_out;
    }

    // If out is None, return error requiring out or caller passes out
    PyBuffer_Release(&buf_a);
    PyBuffer_Release(&buf_b);
    PyErr_SetString(PyExc_NotImplementedError, "Pass pre-allocated out buffer for maximum speed");
    return NULL;
}

static PyObject* py_nanogemm_isa(PyObject* self, PyObject* Py_UNUSED(args)) {
    return PyUnicode_FromString(nanogemm_simd_isa());
}

static PyMethodDef NanoGemmMethods[] = {
    {"matmul_fast", py_nanogemm_matmul, METH_VARARGS, "Fast SIMD GEMM via Python buffer protocol"},
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
    return PyModule_Create(&nanogemm_ext_module);
}
