"""Error detection system for monitoring and identifying issues."""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, informational
    MEDIUM = "medium"     # Concerning but not critical
    HIGH = "high"         # Critical errors requiring immediate attention
    CRITICAL = "critical" # System-breaking errors


class ErrorType(Enum):
    """Types of errors that can be detected."""
    RATE_LIMIT = "rate_limit"           # API rate limit errors
    API_ERROR = "api_error"             # API connection/response errors
    IMPORT_ERROR = "import_error"       # Missing dependencies
    CONFIG_ERROR = "config_error"       # Configuration issues
    GIT_ERROR = "git_error"             # Git operation failures
    ATTRIBUTE_ERROR = "attribute_error" # Code attribute errors
    TYPE_ERROR = "type_error"           # Type-related errors
    SERVICE_CRASH = "service_crash"     # Service stopped/crashed
    TIMEOUT = "timeout"                 # Operation timeouts
    UNKNOWN = "unknown"                 # Unclassified errors


class DetectedError:
    """Represents a detected error."""

    def __init__(
        self,
        error_type: ErrorType,
        severity: ErrorSeverity,
        message: str,
        timestamp: datetime,
        context: Optional[str] = None,
        auto_fixable: bool = False
    ):
        self.error_type = error_type
        self.severity = severity
        self.message = message
        self.timestamp = timestamp
        self.context = context
        self.auto_fixable = auto_fixable

    def __repr__(self):
        return f"<DetectedError type={self.error_type.value} severity={self.severity.value} auto_fixable={self.auto_fixable}>"


class ErrorDetector:
    """Monitors logs and detects errors for auto-fixing."""

    # Error patterns to detect
    ERROR_PATTERNS = {
        ErrorType.RATE_LIMIT: [
            r"429.*rate limit",
            r"RateLimitError",
            r"exceed.*rate limit"
        ],
        ErrorType.API_ERROR: [
            r"API error",
            r"APIError",
            r"500.*Internal Server Error",
            r"503.*Service Unavailable",
            r"Connection.*failed"
        ],
        ErrorType.IMPORT_ERROR: [
            r"ImportError",
            r"ModuleNotFoundError",
            r"No module named"
        ],
        ErrorType.CONFIG_ERROR: [
            r"ConfigError",
            r"configuration.*error",
            r"missing.*config",
            r"invalid.*configuration"
        ],
        ErrorType.GIT_ERROR: [
            r"git.*error",
            r"fatal:.*git",
            r"merge conflict"
        ],
        ErrorType.ATTRIBUTE_ERROR: [
            r"AttributeError",
            r"has no attribute",
            r"NoneType.*has no attribute"
        ],
        ErrorType.TYPE_ERROR: [
            r"TypeError",
            r"unsupported.*type",
            r"expected.*got"
        ],
        ErrorType.TIMEOUT: [
            r"TimeoutError",
            r"timed out",
            r"timeout"
        ]
    }

    def __init__(self, log_file: str = "./data/logs/agent.log"):
        """Initialize error detector.

        Args:
            log_file: Path to log file to monitor
        """
        self.log_file = Path(log_file)
        self.detected_errors: List[DetectedError] = []
        self.error_history: Dict[ErrorType, int] = {}

        logger.info(f"ErrorDetector initialized, monitoring: {log_file}")

    def scan_recent_logs(self, minutes: int = 5) -> List[DetectedError]:
        """Scan recent log entries for errors.

        Args:
            minutes: How many minutes back to scan

        Returns:
            List of detected errors
        """
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return []

        errors = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        try:
            with open(self.log_file, 'r') as f:
                # Read last N lines (more efficient than reading entire file)
                lines = self._tail_file(f, 500)

                for line in lines:
                    # Check if line is recent enough
                    timestamp = self._extract_timestamp(line)
                    if timestamp and timestamp < cutoff_time:
                        continue

                    # Check for error patterns
                    detected = self._detect_error_in_line(line, timestamp)
                    if detected:
                        errors.append(detected)
                        self.detected_errors.append(detected)

                        # Update error history
                        error_type = detected.error_type
                        self.error_history[error_type] = self.error_history.get(error_type, 0) + 1

        except Exception as e:
            logger.error(f"Error scanning logs: {e}")

        if errors:
            logger.warning(f"Detected {len(errors)} errors in last {minutes} minutes")

        return errors

    def _detect_error_in_line(self, line: str, timestamp: Optional[datetime]) -> Optional[DetectedError]:
        """Detect error in a single log line.

        Args:
            line: Log line to analyze
            timestamp: Timestamp of the log entry

        Returns:
            DetectedError if found, None otherwise
        """
        line_lower = line.lower()

        # Check if line contains error indicator
        if not any(keyword in line_lower for keyword in ["error", "exception", "failed", "critical"]):
            return None

        # Match against error patterns
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Determine severity and auto-fixability
                    severity, auto_fixable = self._assess_error(error_type, line)

                    return DetectedError(
                        error_type=error_type,
                        severity=severity,
                        message=line.strip(),
                        timestamp=timestamp or datetime.now(),
                        context=self._extract_context(line),
                        auto_fixable=auto_fixable
                    )

        # Unknown error type
        if "error" in line_lower or "exception" in line_lower:
            return DetectedError(
                error_type=ErrorType.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                message=line.strip(),
                timestamp=timestamp or datetime.now(),
                auto_fixable=False
            )

        return None

    def _assess_error(self, error_type: ErrorType, message: str) -> tuple[ErrorSeverity, bool]:
        """Assess error severity and auto-fixability.

        Args:
            error_type: Type of error
            message: Error message

        Returns:
            Tuple of (severity, auto_fixable)
        """
        # Define which errors are auto-fixable
        auto_fixable_types = {
            ErrorType.RATE_LIMIT,      # Can switch to fallback
            ErrorType.IMPORT_ERROR,    # Can install packages
            ErrorType.GIT_ERROR,       # Can auto-resolve conflicts
            ErrorType.ATTRIBUTE_ERROR, # Can sometimes fix code
            ErrorType.CONFIG_ERROR     # Can reset configs
        }

        # Determine severity
        severity_map = {
            ErrorType.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorType.API_ERROR: ErrorSeverity.HIGH,
            ErrorType.IMPORT_ERROR: ErrorSeverity.HIGH,
            ErrorType.CONFIG_ERROR: ErrorSeverity.HIGH,
            ErrorType.GIT_ERROR: ErrorSeverity.MEDIUM,
            ErrorType.ATTRIBUTE_ERROR: ErrorSeverity.HIGH,
            ErrorType.TYPE_ERROR: ErrorSeverity.HIGH,
            ErrorType.SERVICE_CRASH: ErrorSeverity.CRITICAL,
            ErrorType.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
        }

        severity = severity_map.get(error_type, ErrorSeverity.MEDIUM)
        auto_fixable = error_type in auto_fixable_types

        return severity, auto_fixable

    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line.

        Args:
            line: Log line

        Returns:
            Datetime if found, None otherwise
        """
        # Try to match common log timestamp formats
        patterns = [
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",  # 2024-01-01 12:00:00
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"   # ISO format
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    return datetime.fromisoformat(timestamp_str.replace(" ", "T"))
                except:
                    pass

        return None

    def _extract_context(self, line: str) -> Optional[str]:
        """Extract relevant context from error line.

        Args:
            line: Log line

        Returns:
            Context string if found
        """
        # Extract file/line number if present
        file_match = re.search(r'File "([^"]+)", line (\d+)', line)
        if file_match:
            return f"{file_match.group(1)}:{file_match.group(2)}"

        # Extract module name if present
        module_match = re.search(r'in (\w+\.py)', line)
        if module_match:
            return module_match.group(1)

        return None

    def _tail_file(self, file_handle, n: int = 500) -> List[str]:
        """Read last N lines from file efficiently.

        Args:
            file_handle: Open file handle
            n: Number of lines to read

        Returns:
            List of last N lines
        """
        # Simple implementation - for large files, could use more efficient method
        lines = file_handle.readlines()
        return lines[-n:] if len(lines) > n else lines

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of detected errors.

        Returns:
            Summary dict with error statistics
        """
        total = len(self.detected_errors)
        auto_fixable = sum(1 for e in self.detected_errors if e.auto_fixable)

        by_type = {}
        for error_type, count in self.error_history.items():
            by_type[error_type.value] = count

        by_severity = {}
        for error in self.detected_errors:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_errors": total,
            "auto_fixable": auto_fixable,
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_errors": [
                {
                    "type": e.error_type.value,
                    "severity": e.severity.value,
                    "message": e.message[:100],  # Truncate
                    "auto_fixable": e.auto_fixable
                }
                for e in self.detected_errors[-5:]  # Last 5 errors
            ]
        }

    def clear_history(self):
        """Clear error history."""
        self.detected_errors.clear()
        self.error_history.clear()
        logger.info("Error history cleared")
