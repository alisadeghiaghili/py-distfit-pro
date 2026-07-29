import pytest
import numpy as np


@pytest.fixture
def normal_data():
    np.random.seed(42)
    return np.random.normal(10, 2, 500)


@pytest.fixture
def exponential_data():
    np.random.seed(42)
    return np.random.exponential(2, 500)


@pytest.fixture
def poisson_data():
    np.random.seed(42)
    return np.random.poisson(5, 500)
