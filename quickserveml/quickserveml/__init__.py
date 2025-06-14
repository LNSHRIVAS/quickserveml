"""
QuickServeML - One-command ONNX model serving, benchmarking, and visual inspection.

A comprehensive CLI tool that transforms ONNX models into production-ready APIs
with just one command. Provides essential features for ML model analysis,
optimization, and deployment.
"""

__version__ = "1.0.2"
__author__ = "Lakshminarayan Shrivas"
__email__ = "lakshminarayanshrivas7@gmail.com"
__description__ = "One-command ONNX model serving, benchmarking, and visual inspection"
__url__ = "https://github.com/LNSHRIVAS/quickserveml"
__license__ = "MIT"

# Import main CLI interface
from .cli import cli

__all__ = ["cli"]
