from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="distfit-pro",
    version="1.0.0",
    author="Ali Sadeghi Aghili",
    author_email="alisadeghiaghili@gmail.com",
    description="Professional distribution fitting library with 25 distributions, GOF tests, bootstrap CI, and multilingual support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alisadeghiaghili/py-distfit-pro",
    project_urls={
        "Documentation": "https://github.com/alisadeghiaghili/py-distfit-pro/docs",
        "Source": "https://github.com/alisadeghiaghili/py-distfit-pro",
        "Tracker": "https://github.com/alisadeghiaghili/py-distfit-pro/issues",
        "Changelog": "https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Statistics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
        "Natural Language :: English",
        "Natural Language :: Persian",
        "Natural Language :: German",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core numerical
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        
        # Visualization
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # Parallel processing
        "joblib>=1.0.0",
        
        # Progress bars
        "tqdm>=4.60.0",
        
        # RTL text support (Persian/Arabic/Hebrew)
        "arabic-reshaper>=2.1.0",
        "python-bidi>=0.4.2",
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx_rtd_theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        # Optional statistical tests
        "stats": [
            "statsmodels>=0.12.0",
        ],
        # All optional dependencies
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx_rtd_theme>=1.0.0",
            "myst-parser>=0.18.0",
            "statsmodels>=0.12.0",
        ],
    },
    include_package_data=True,
    package_data={
        "distfit_pro": [
            "locales/*.json",
            "locales/**/*.json",
        ],
    },
    keywords=[
        # Core functionality
        "statistics",
        "distribution fitting",
        "probability distributions",
        "parameter estimation",
        
        # Methods
        "maximum likelihood",
        "method of moments",
        "goodness of fit",
        "bootstrap",
        "confidence intervals",
        
        # Advanced features
        "diagnostics",
        "weighted data",
        "model selection",
        "AIC",
        "BIC",
        
        # Features
        "multilingual",
        "visualization",
        "data science",
        "statistical analysis",
        "data analysis",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # Add CLI tools if needed in future
        ],
    },
)
