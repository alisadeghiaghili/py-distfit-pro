# Contributing to DistFit Pro

Thank you for your interest in contributing! 🎉

## Ways to Contribute

### 1. Report Bugs

**Before submitting:**
- Check if already reported
- Include minimal reproducible example
- Specify Python version, OS, package version

**Create issue with:**
```python
# Code that reproduces the bug
import distfit_pro
# ...
```

### 2. Request Features

**Good feature requests include:**
- Use case / motivation
- Example API
- Why existing features don't work

### 3. Fix Bugs

Look for issues labeled `bug` or `good-first-issue`.

### 4. Add Features

**Before starting:**
1. Open issue to discuss
2. Get approval from maintainer
3. Follow coding standards

### 5. Improve Documentation

- Fix typos
- Add examples
- Clarify explanations
- Translate to new languages

## Development Setup

### Clone & Install

```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e ".[dev]"
```

### Development Dependencies

```bash
pip install pytest pytest-cov black isort mypy sphinx
```

## Coding Standards

### Style

- **Format:** Black (line length 100)
- **Imports:** isort
- **Type hints:** Required for public APIs
- **Docstrings:** NumPy style

**Format code:**
```bash
black distfit_pro/
isort distfit_pro/
```

### Documentation

**Every public function needs:**

```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description.
    
    Longer explanation if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
        
    Returns
    -------
    result : float
        Description of return value
        
    Examples
    --------
    >>> my_function(42, "test")
    3.14
    """
    pass
```

### Testing

**All new code needs tests:**

```python
# tests/test_myfeature.py
import pytest
from distfit_pro import my_function

def test_my_function_basic():
    result = my_function(42, "test")
    assert result > 0

def test_my_function_edge_case():
    with pytest.raises(ValueError):
        my_function(-1, "invalid")
```

**Run tests:**
```bash
pytest tests/
pytest --cov=distfit_pro tests/  # with coverage
```

## Pull Request Process

### 1. Create Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation
- Run tests locally

### 3. Commit

**Commit message format:**
```
type: short description

Longer explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring
- `perf`: Performance improvement

**Example:**
```
feat: add Nakagami distribution

Implemented Nakagami distribution with MLE and moments fitting.
Added comprehensive tests and documentation.

Fixes #45
```

### 4. Push & Create PR

```bash
git push origin feature/my-new-feature
```

Then create PR on GitHub.

**PR should include:**
- Description of changes
- Related issue number
- Tests added/updated
- Documentation updated

### 5. Code Review

- Address reviewer comments
- Update PR
- Get approval

### 6. Merge

Maintainer will merge after approval.

## Adding a New Distribution

### Step-by-Step

**1. Create distribution class:**

```python
# distfit_pro/core/distributions.py

class MyDistribution(BaseDistribution):
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="mydist",
            display_name="My Distribution",
            parameters={
                "alpha": "Shape parameter (α > 0)",
                "beta": "Scale parameter (β > 0)"
            },
            support="x > 0",
            use_cases=[
                "Use case 1",
                "Use case 2"
            ],
            characteristics=[
                "Characteristic 1",
                "Characteristic 2"
            ]
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        # Implement PDF
        pass
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        # Implement CDF
        pass
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        # Implement inverse CDF
        pass
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        # Implement MLE
        pass
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        # Implement method of moments
        pass
```

**2. Register distribution:**

```python
# distfit_pro/core/distributions.py

DISTRIBUTION_REGISTRY = {
    # ...
    'mydist': MyDistribution,
}
```

**3. Add tests:**

```python
# tests/test_distributions.py

def test_mydist_pdf():
    dist = get_distribution('mydist')
    dist.params = {'alpha': 2.0, 'beta': 1.0}
    # Test PDF

def test_mydist_fit():
    data = # Generate test data
    dist = get_distribution('mydist')
    dist.fit(data)
    # Verify parameters
```

**4. Add documentation:**

```rst
# docs/source/api/distributions.rst

My Distribution
^^^^^^^^^^^^^^^

Description...
```

**5. Add translations:**

```python
# distfit_pro/locales/fa.py (for Farsi)
# Add translations
```

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to PyPI
5. Create GitHub release

## Questions?

Open an issue or contact:
- [@alisadeghiaghili](https://github.com/alisadeghiaghili)

## Code of Conduct

Be respectful, inclusive, and professional.

Thank you for contributing! 🙏
