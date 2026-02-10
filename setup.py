"""
DistFit Pro - Professional Distribution Fitting Package
========================================================

A comprehensive, production-ready Python package for statistical distribution fitting
that combines the best features of EasyFit and fitdistrplus with modern improvements.

Author: Ali Aghili (https://zil.ink/thedatascientist)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="distfit-pro",
    version="0.1.0",
    author="Ali Aghili",
    author_email="alisadeghiaghili@gmail.com",
    description="Professional distribution fitting with model selection, diagnostics, and explanations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alisadeghiaghili/py-distfit-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)