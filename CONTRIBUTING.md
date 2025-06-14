# Contributing to QuickServeML 🤝

Thank you for your interest in contributing to QuickServeML! This document provides guidelines and information for contributors.

## 🎯 **How Can I Contribute?**

### **Reporting Bugs**
- Use the [GitHub Issues](https://github.com/LNSHRIVAS/quickserveml/issues) page
- Include detailed steps to reproduce the bug
- Provide system information (OS, Python version, etc.)
- Include error messages and stack traces

### **Suggesting Enhancements**
- Use the [GitHub Issues](https://github.com/LNSHRIVAS/quickserveml/issues) page
- Describe the feature and its benefits
- Provide use cases and examples
- Consider implementation complexity

### **Code Contributions**
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests
- Submit a pull request

## 🛠️ **Development Setup**

### **Prerequisites**
- Python 3.7 or higher
- Git
- pip

### **Local Development Setup**
```bash
# Clone the repository
git clone https://github.com/LNSHRIVAS/quickserveml.git
cd quickserveml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### **Testing Your Setup**
```bash
# Test the CLI
quickserveml --help

# Run the comprehensive demo
python examples/comprehensive_demo.py mnist-8.onnx
```

## 📝 **Code Style and Standards**

### **Python Code Style**
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints for function parameters and return values

### **Running Code Quality Tools**
```bash
# Format code with Black
black quickserveml/ examples/

# Check code style with flake8
flake8 quickserveml/ examples/

# Type checking with mypy
mypy quickserveml/
```

### **Documentation Standards**
- Use docstrings for all public functions and classes
- Follow [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring format
- Include type hints in docstrings
- Provide examples for complex functions

### **Example Docstring**
```python
def benchmark_model(model_path: str, 
                   warmup_runs: int = 10,
                   benchmark_runs: int = 100,
                   provider: str = "CPUExecutionProvider",
                   verbose: bool = True) -> BenchmarkResult:
    """Benchmark an ONNX model and return performance metrics.
    
    Args:
        model_path: Path to the ONNX model file
        warmup_runs: Number of warmup runs to ensure consistent performance
        benchmark_runs: Number of benchmark runs for statistical accuracy
        provider: ONNX Runtime execution provider (CPU/GPU)
        verbose: Whether to print progress and results
    
    Returns:
        BenchmarkResult: Object containing all performance metrics
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
        
    Example:
        >>> result = benchmark_model("model.onnx", verbose=True)
        >>> print(f"Throughput: {result.throughput_rps:.1f} req/s")
    """
```

## 🧪 **Testing**

### **Running Tests**
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=quickserveml --cov-report=html

# Run specific test file
pytest tests/test_benchmark.py

# Run tests in verbose mode
pytest -v
```

### **Writing Tests**
- Create tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Mock external dependencies

### **Example Test**
```python
import pytest
from quickserveml.benchmark import benchmark_model

def test_benchmark_model_success():
    """Test successful model benchmarking."""
    result = benchmark_model("test_model.onnx", verbose=False)
    assert result.avg_inference_time_ms > 0
    assert result.throughput_rps > 0
    assert result.memory_usage_mb > 0

def test_benchmark_model_file_not_found():
    """Test benchmarking with non-existent model."""
    with pytest.raises(FileNotFoundError):
        benchmark_model("nonexistent_model.onnx")
```

## 🔄 **Contribution Workflow**

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/quickserveml.git
cd quickserveml

# Add upstream remote
git remote add upstream https://github.com/LNSHRIVAS/quickserveml.git
```

### **2. Create Feature Branch**
```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### **3. Make Changes**
- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation if needed
- Commit your changes with descriptive messages

### **4. Commit Messages**
Use clear, descriptive commit messages:
```bash
# Good commit messages
git commit -m "Add batch processing optimization feature"
git commit -m "Fix memory leak in benchmark function"
git commit -m "Update documentation for new CLI options"

# Avoid vague messages
git commit -m "Fix stuff"
git commit -m "Update"
```

### **5. Push and Create Pull Request**
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### **6. Pull Request Guidelines**
- Provide a clear description of changes
- Include any relevant issue numbers
- Add screenshots for UI changes
- Ensure all tests pass
- Update documentation if needed

## 🏗️ **Project Structure**

```
quickserveml/
├── quickserveml/           # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── infer.py            # Model inspection
│   ├── benchmark.py        # Performance benchmarking
│   ├── schema.py           # Schema generation
│   ├── batch.py            # Batch processing
│   └── serve.py            # FastAPI server
├── templates/              # Jinja templates
├── examples/               # Example scripts
├── tests/                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # Project documentation
```

## 🐛 **Common Issues and Solutions**

### **Import Errors**
```bash
# If you get import errors, ensure you're in the virtual environment
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .
```

### **ONNX Runtime Issues**
```bash
# If ONNX Runtime fails to load models
pip install --upgrade onnxruntime
```

### **Template Not Found**
```bash
# If templates are not found during deployment
# Ensure you're running from the project root directory
cd /path/to/quickserveml
quickserveml deploy model.onnx
```

## 📚 **Additional Resources**

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Click Documentation](https://click.palletsprojects.com/)

## 🤝 **Getting Help**

- **GitHub Issues**: [Create an issue](https://github.com/LNSHRIVAS/quickserveml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LNSHRIVAS/quickserveml/discussions)
- **Email**: [your-email@example.com]

## 📄 **License**

By contributing to QuickServeML, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to QuickServeML! 🚀 