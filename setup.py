import os
import sys
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

if sys.platform == "darwin":
    machine = platform.machine().lower()
    if "arm" in machine:
        os.environ.setdefault("ARCHFLAGS", "-arch arm64")
    elif "x86" in machine:
        os.environ.setdefault("ARCHFLAGS", "-arch x86_64")


class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        machine = platform.machine().lower()
        is_arm = machine in ("arm64", "aarch64") or ("arm" in machine)

        for ext in self.extensions:
            if compiler_type == "msvc":
                if is_arm:
                    ext.extra_compile_args = ["/O2", "/fp:fast", "/DBUILDING_NANOGEMM"]
                else:
                    ext.extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast", "/DBUILDING_NANOGEMM"]
            else:
                if is_arm:
                    ext.extra_compile_args = ["-O3", "-ffast-math", "-fPIC", "-DBUILDING_NANOGEMM"]
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