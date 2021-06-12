from setuptools import setup
from Cython.Build import cythonize

setup(
    name="matrixcomp_c",
    version="0.0.1",
    url="https://github.com/choct155/matrixComputations",
    ext_modules=cythonize("matrixcomp_c.pyx"),
    zip_safe=False
)