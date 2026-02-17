"""Self-healing components for autonomous error detection and recovery."""

from .error_detector import ErrorDetector
from .auto_fixer import AutoFixer

__all__ = ["ErrorDetector", "AutoFixer"]
