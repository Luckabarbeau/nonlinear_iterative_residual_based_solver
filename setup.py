
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("nonlinear_gmres_gmg_mf.pyx")
)
