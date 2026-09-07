import sys
from setuptools import setup, Extension, find_packages

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith("win"):
    extra_compile_args = ["-O3", "-mavx2", "-mfma", "-ffast-math", "-DBUILDING_NANOGEMM"]
else:
    extra_compile_args = ["-O3", "-mavx2", "-mfma", "-ffast-math", "-fPIC", "-DBUILDING_NANOGEMM"]

ext_modules = [
    Extension(
        "nanogemm._ext",
        sources=["src/nanogemm_pyext.c", "src/nanogemm_kernel.c"],
        include_dirs=["src"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="nanogemm",
    packages=find_packages(),
    ext_modules=ext_modules,
    package_data={"nanogemm": ["*.dll", "*.so", "*.dylib"]},
)
