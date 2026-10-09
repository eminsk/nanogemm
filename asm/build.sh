#!/usr/bin/env bash
# =============================================================================
# NanoGEMM — Cross-Platform Native Library Builder for Linux & macOS
# Compiles hardware-accelerated SIMD GEMM engines (AVX2+FMA / ARM NEON)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_KERNEL="${ROOT_DIR}/src/nanogemm_kernel.c"
OUT_DIR="${ROOT_DIR}/nanogemm"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "====================================================================="
echo "  NanoGEMM - Native Library Builder"
echo "  Detected OS:   ${OS}"
echo "  Architecture:  ${ARCH}"
echo "====================================================================="

mkdir -p "${OUT_DIR}"

if [ "${OS}" = "Darwin" ]; then
    # macOS (Apple Silicon M1/M2/M3/M4 or Intel x86_64)
    OUT_LIB="${OUT_DIR}/libnanogemm.dylib"
    echo "Compiling for macOS using Apple Clang..."
    if [ "${ARCH}" = "arm64" ]; then
        clang -O3 -shared -fPIC -ffast-math -DBUILDING_NANOGEMM \
            -arch arm64 "${SRC_KERNEL}" -o "${OUT_LIB}"
    elif [ "${ARCH}" = "x86_64" ]; then
        clang -O3 -shared -fPIC -mavx2 -mfma -ffast-math -DBUILDING_NANOGEMM \
            -arch x86_64 "${SRC_KERNEL}" -o "${OUT_LIB}"
    else
        clang -O3 -shared -fPIC -ffast-math -DBUILDING_NANOGEMM \
            "${SRC_KERNEL}" -o "${OUT_LIB}"
    fi
    echo "[OK] Built macOS dynamic library: ${OUT_LIB}"

elif [ "${OS}" = "Linux" ]; then
    # Linux (x86_64 or aarch64)
    OUT_LIB="${OUT_DIR}/libnanogemm.so"
    echo "Compiling for Linux using GCC/Clang..."
    CC=${CC:-gcc}
    if [ "${ARCH}" = "x86_64" ]; then
        ${CC} -O3 -shared -fPIC -mavx2 -mfma -ffast-math -DBUILDING_NANOGEMM \
            "${SRC_KERNEL}" -o "${OUT_LIB}"
        
        # Optional FASM assembly build if fasm binary is installed
        if command -v fasm &> /dev/null && [ -f "${SCRIPT_DIR}/nanogemm64_linux.asm" ]; then
            echo "Building with Flat Assembler (FASM)..."
            fasm "${SCRIPT_DIR}/nanogemm64_linux.asm" "${OUT_DIR}/nanogemm64.so" || true
        fi
    elif [ "${ARCH}" = "aarch64" ]; then
        ${CC} -O3 -shared -fPIC -ffast-math -DBUILDING_NANOGEMM \
            "${SRC_KERNEL}" -o "${OUT_LIB}"
    else
        ${CC} -O3 -shared -fPIC -ffast-math -DBUILDING_NANOGEMM \
            "${SRC_KERNEL}" -o "${OUT_LIB}"
    fi
    echo "[OK] Built Linux shared object: ${OUT_LIB}"

else
    echo "[ERROR] Unsupported operating system: ${OS}"
    exit 1
fi

echo "====================================================================="
echo "  Build Complete! NanoGEMM native acceleration is ready."
echo "====================================================================="
