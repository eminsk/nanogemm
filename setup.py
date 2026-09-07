import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast", "/DBUILDING_NANOGEMM"]
            else:
                ext.extra_compile_args = ["-O3", "-mavx2", "-mfma", "-ffast-math", "-fPIC", "-DBUILDING_NANOGEMM"]
        super().build_extensions()

ext_modules = [
    Extension(
        "nanogemm._ext",
        sources=["src/nanogemm_pyext.c", "src/nanogemm_kernel.c"],
        include_dirs=["src"],
    )
]

setup(
    name="nanogemm",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    package_data={"nanogemm": ["*.dll", "*.so", "*.dylib"]},
)