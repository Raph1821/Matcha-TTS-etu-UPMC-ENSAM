"""
Script simple pour compiler core.pyx en extension Cython
"""
import os
import sys
from pathlib import Path
import numpy as np

try:
    from Cython.Build import cythonize
    from setuptools import Extension, setup
except ImportError:
    print(" Cython n'est pas installé. Installez-le avec: pip install Cython")
    sys.exit(1)

# Chemin vers core.pyx
project_root = Path(__file__).parent
core_pyx = project_root / "matcha" / "utils" / "monotonic_align" / "core.pyx"

if not core_pyx.exists():
    print(f" Fichier introuvable: {core_pyx}")
    sys.exit(1)

print(f" Compilation de: {core_pyx}")

# Obtenir le chemin des headers numpy
numpy_include = np.get_include()

# Configuration de l'extension
extensions = [
    Extension(
        "matcha.utils.monotonic_align.core",
        [str(core_pyx)],
        include_dirs=[numpy_include],  # Ajouter le chemin numpy
        extra_compile_args=["-fopenmp"] if os.name != 'nt' else [],
        extra_link_args=["-fopenmp"] if os.name != 'nt' else [],
        language="c",
    )
]

# Compilation
extensions = cythonize(
    extensions,
    compiler_directives={"language_level": "3"},
    build_dir="build"
)

# Compiler
setup(
    name="matcha-cython-ext",
    ext_modules=extensions,
    script_args=["build_ext", "--inplace"]
)

print("\n Compilation terminée!")
print(" Le fichier .so devrait être dans: matcha/utils/monotonic_align/")

