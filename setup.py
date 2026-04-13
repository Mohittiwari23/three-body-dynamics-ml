"""
setup.py
=========
Install the full orbital stability research framework:

    pip install -e .                # simulation + ML
    pip install -e ".[dev]"         # + test suite
    pip install -e ".[notebook]"    # + Jupyter
"""

from setuptools import setup, find_packages

setup(
    name        = "orbital_simulator",
    version     = "0.2.0",
    description = "Physics-driven orbital dynamics simulator and ML stability framework",
    author      = "Orbital Simulator Project",
    python_requires = ">=3.10",
    packages    = find_packages(),      # finds orbital_simulator/ and ml/
    install_requires = [
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "scipy>=1.11",
    ],
    extras_require = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "notebook": [
            "jupyter",
            "ipympl",
        ],
    },
    entry_points = {
        "console_scripts": [
            "orbital-sim=main:main",
        ],
    },
)