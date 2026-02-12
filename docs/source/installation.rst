Installation
============

Requirements
------------

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3
- Plotly >= 5.0
- joblib >= 1.0
- tqdm >= 4.60

Install from PyPI
-----------------

.. code-block:: bash

    pip install distfit-pro

Install from Source
-------------------

.. code-block:: bash

    git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
    cd py-distfit-pro
    pip install -e .

Verify Installation
-------------------

.. code-block:: python

    import distfit_pro
    print(distfit_pro.__version__)
    
    from distfit_pro import get_distribution, list_distributions
    print(f"Available distributions: {len(list_distributions())}")

Development Installation
------------------------

For contributors:

.. code-block:: bash

    git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
    cd py-distfit-pro
    pip install -e ".[dev]"
    
    # Run tests
    pytest tests/
    
    # Build documentation
    cd docs
    make html
