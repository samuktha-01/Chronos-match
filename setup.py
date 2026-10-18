"""Build configuration for ChronosMatch Cython extension modules."""

from __future__ import annotations

import os
from pathlib import Path
from setuptools import Extension, setup

# Optimize compiler flags for Windows MSVC vs GCC/Clang
extra_compile_args: list[str] = (
    ["/O2", "/fp:fast"] if os.name == "nt" else ["-O3", "-march=native"]
)

pyx_file = Path("src") / "chronosmatch" / "_cython_engine.pyx"
c_file = Path("src") / "chronosmatch" / "_cython_engine.c"

ext_modules = []
try:
    from Cython.Build import cythonize

    source_file = str(pyx_file) if pyx_file.exists() else str(c_file)
    ext_modules = cythonize(
        [
            Extension(
                "chronosmatch._cython_engine",
                sources=[source_file],
                extra_compile_args=extra_compile_args,
            )
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    if c_file.exists():
        ext_modules = [
            Extension(
                "chronosmatch._cython_engine",
                sources=[str(c_file)],
                extra_compile_args=extra_compile_args,
            )
        ]

if __name__ == "__main__":
    setup(ext_modules=ext_modules)
