# Contributing to arXiv Researcher

Thank you for your interest in contributing to arXiv Researcher! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a tool to help researchers, so let's maintain a collaborative and welcoming environment.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/dcondrey/arXivResearcher/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Your environment (OS, Python version, package version)
   - Any relevant error messages or logs

### Suggesting Features

1. Check if the feature has been suggested in [Issues](https://github.com/dcondrey/arXivResearcher/issues)
2. Create a new issue with the "enhancement" label including:
   - Clear description of the feature
   - Use case / motivation
   - Proposed implementation (optional)

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run linting: `ruff check src/ tests/`
6. Commit with a clear message
7. Push and create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/arXivResearcher.git
cd arXivResearcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Google-style** docstrings

### Example

```python
def analyze_trends(
    papers: list[dict],
    window_size: int = 30,
    min_count: int = 5,
) -> dict[str, list[str]]:
    """Analyze topic trends over time.

    Args:
        papers: List of paper dictionaries with 'published_date' and 'keywords'.
        window_size: Rolling window size in days for trend calculation.
        min_count: Minimum occurrences to be considered a trend.

    Returns:
        Dictionary with 'emerging' and 'declining' topic lists.

    Raises:
        ValueError: If papers list is empty.
    """
    if not papers:
        raise ValueError("Papers list cannot be empty")

    # Implementation...
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=arxiv_researcher --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run specific test
pytest tests/test_analyzer.py::test_trend_analysis
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for >80% coverage on new code

## Documentation

- Update docstrings for any changed functions
- Update README.md if adding new features
- Add examples for new functionality

## Commit Messages

Use clear, descriptive commit messages:

```
Add semantic embedding analyzer for paper clustering

- Implement EmbeddingAnalyzer class with sentence-transformers
- Add UMAP dimensionality reduction for visualization
- Add clustering with HDBSCAN
- Include similarity search functionality
```

## Release Process

1. Update version in `pyproject.toml` and `src/arxiv_researcher/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v1.x.x`
4. Push: `git push origin main --tags`
5. GitHub Actions will handle PyPI release

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in the Discussions tab
- Email david@writerslogic.com

Thank you for contributing!
