# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:53:52 2024

@author: lucka
"""
import os
import subprocess
import sys

# Step 1: Create the `setup.py` script for the provided `.pyx` file
setup_code = """
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("nonlinear_gmres_gmg_mf.pyx")
)
"""

with open("setup.py", "w") as f:
    f.write(setup_code)

# Step 2: Compile the `.pyx` file using setup.py
# This will create a shared object file (.so or .pyd) that can be imported in Python
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"])

# Step 3: Import the compiled module and run a function
try:
    # Importing the compiled module
    import nonlinear_gmres_gmg_mf

    # Example call (this assumes your module has a callable function named `main_function`)
    # Replace `main_function` with the appropriate function name in your `.pyx` file.
    if hasattr(nonlinear_gmres_gmg_mf, 'main_function'):
        result = nonlinear_gmres_gmg_mf.main_function()  # Adjust arguments as needed
        print(f"Result of the main function: {result}")
    else:
        print("Compilation successful, but no callable function `main_function` found in the module.")

except ImportError as e:
    print(f"Error importing the compiled module: {e}")

# Cleanup generated files if desired (optional)
cleanup_files = [ "nonlinear_gmres_gmg_mf.c"]
for filename in cleanup_files:
    if os.path.exists(filename):
        os.remove(filename)