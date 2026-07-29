import os
from pathlib import Path

from setuptools import Extension, setup


gmp_prefixes = [
    Path(os.environ["SPONG_GMP_PREFIX"])
    for _ in (0,) if os.environ.get("SPONG_GMP_PREFIX")
]
gmp_prefixes += [Path("/opt/homebrew"), Path("/usr/local")]
gmp_include_dirs = [
    str(prefix / "include") for prefix in gmp_prefixes
    if (prefix / "include" / "gmp.h").exists()
]
gmp_library_dirs = [
    str(prefix / "lib") for prefix in gmp_prefixes
    if any((prefix / "lib").glob("libgmp.*"))
]


setup(
    ext_modules=[
        Extension(
            "spong._native",
            ["src/spong/_native.c", "src/c/spong_resolution.c",
             "src/c/spong_exact_gmp.c"],
            include_dirs=["include", *gmp_include_dirs],
            library_dirs=gmp_library_dirs,
            libraries=["gmp"],
            extra_compile_args=["-O3"],
        )
    ],
)
