from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "spong._native",
            ["src/spong/_native.c"],
            extra_compile_args=["-O3"],
        )
    ],
)
