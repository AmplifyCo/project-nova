"""Audit logging for security-sensitive operations.

This module provides comprehensive audit logging for:
- Bash command execution
- File operations (read, write, delete)
- Security violations (prompt injection, rate limiting, etc.)
- Tool usage
- API calls
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """Security audit logger for tracking sensitive operations."""

    def __init__(self, audit_log_path: str = "logs/security_audit.jsonl"):
        """Initialize audit logger.

        Args:
            audit_log_path: Path to audit log file (JSONL format)
        """
        self.audit_log_path = Path(audit_log_path)

        # Ensure log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audit logging enabled: {self.audit_log_path}")

    def _write_audit_entry(self, entry: Dict[str, Any]):
        """Write audit entry to log file.

        Args:
            entry: Audit entry dictionary
        """
        try:
            # Add timestamp
            entry["timestamp"] = datetime.now().isoformat()

            # Write as JSONL (one JSON object per line)
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    # ========================================================================
    # Security Events
    # ========================================================================

    def log_security_violation(
        self,
        violation_type: str,
        user_id: str,
        channel: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security violation (prompt injection, data extraction, etc.).

        Args:
            violation_type: Type of violation (prompt_injection, data_extraction, etc.)
            user_id: User identifier
            channel: Communication channel
            message: User message that triggered violation
            details: Additional details
        """
        entry = {
            "event_type": "security_violation",
            "severity": "critical",
            "violation_type": violation_type,
            "user_id": user_id,
            "channel": channel,
            "message": message[:200],  # Truncate to avoid huge logs
            "details": details or {}
        }

        self._write_audit_entry(entry)
        logger.warning(f"🚨 AUDIT: Security violation - {violation_type} from {user_id}")

    def log_rate_limit_exceeded(
        self,
        user_id: str,
        channel: str,
        request_count: int,
        window_seconds: int
    ):
        """Log rate limit violation.

        Args:
            user_id: User identifier
            channel: Communication channel
            request_count: Number of requests in window
            window_seconds: Time window in seconds
        """
        entry = {
            "event_type": "rate_limit_exceeded",
            "severity": "warning",
            "user_id": user_id,
            "channel": channel,
            "request_count": request_count,
            "window_seconds": window_seconds
        }

        self._write_audit_entry(entry)
        logger.warning(f"🚨 AUDIT: Rate limit exceeded - {user_id} ({request_count} requests)")

    # ========================================================================
    # Tool Execution
    # ========================================================================

    def log_bash_command(
        self,
        command: str,
        user_id: str,
        success: bool,
        output: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log bash command execution.

        Args:
            command: Bash command executed
            user_id: User identifier
            success: Whether command succeeded
            output: Command output (truncated)
            error: Error message if failed
        """
        entry = {
            "event_type": "bash_command",
            "severity": "info",
            "command": command,
            "user_id": user_id,
            "success": success,
            "output": output[:500] if output else None,  # Truncate output
            "error": error
        }

        self._write_audit_entry(entry)

        if not success:
            logger.info(f"🚨 AUDIT: Bash command failed - {command[:100]}")

    def log_file_operation(
        self,
        operation: str,
        file_path: str,
        user_id: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log file operation (read, write, delete).

        Args:
            operation: Operation type (read, write, delete)
            file_path: File path
            user_id: User identifier
            success: Whether operation succeeded
            details: Additional details
        """
        entry = {
            "event_type": "file_operation",
            "severity": "info" if operation == "read" else "warning",
            "operation": operation,
            "file_path": file_path,
            "user_id": user_id,
            "success": success,
            "details": details or {}
        }

        self._write_audit_entry(entry)

        if operation in ["write", "delete"]:
            logger.info(f"🚨 AUDIT: File {operation} - {file_path}")

    def log_tool_execution(
        self,
        tool_name: str,
        user_id: str,
        user_message: str,
        params: Dict[str, Any],
        success: bool,
        validation_result: Optional[str] = None
    ):
        """Log tool execution with semantic validation result.

        Args:
            tool_name: Tool name
            user_id: User identifier
            user_message: Original user message
            params: Tool parameters
            success: Whether execution succeeded
            validation_result: Semantic validation result (if applicable)
        """
        entry = {
            "event_type": "tool_execution",
            "severity": "info",
            "tool_name": tool_name,
            "user_id": user_id,
            "user_message": user_message[:200],
            "params": params,
            "success": success,
            "validation_result": validation_result
        }

        self._write_audit_entry(entry)

    # ========================================================================
    # API Calls
    # ========================================================================

    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        user_id: str,
        success: bool,
        status_code: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Log external API call.

        Args:
            api_name: API name (e.g., "claude", "openai")
            endpoint: API endpoint
            user_id: User identifier
            success: Whether call succeeded
            status_code: HTTP status code
            error: Error message if failed
        """
        entry = {
            "event_type": "api_call",
            "severity": "info",
            "api_name": api_name,
            "endpoint": endpoint,
            "user_id": user_id,
            "success": success,
            "status_code": status_code,
            "error": error
        }

        self._write_audit_entry(entry)

    # ========================================================================
    # Sensitive Data Access
    # ========================================================================

    def log_sensitive_data_access(
        self,
        data_type: str,
        user_id: str,
        access_granted: bool,
        reason: Optional[str] = None
    ):
        """Log access to sensitive data (API keys, credentials, etc.).

        Args:
            data_type: Type of sensitive data (api_key, password, etc.)
            user_id: User identifier
            access_granted: Whether access was granted
            reason: Reason for access/denial
        """
        entry = {
            "event_type": "sensitive_data_access",
            "severity": "critical" if access_granted else "warning",
            "data_type": data_type,
            "user_id": user_id,
            "access_granted": access_granted,
            "reason": reason
        }

        self._write_audit_entry(entry)
        logger.warning(f"🚨 AUDIT: Sensitive data access - {data_type} ({user_id})")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        severity: Optional[str] = None
    ) -> list:
        """Get recent audit events.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            severity: Filter by severity

        Returns:
            List of audit events
        """
        try:
            events = []

            if not self.audit_log_path.exists():
                return events

            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        if event_type and event.get("event_type") != event_type:
                            continue
                        if severity and event.get("severity") != severity:
                            continue

                        events.append(event)

                    except json.JSONDecodeError:
                        continue

            # Return most recent events
            return events[-limit:]

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []

    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events.

        Returns:
            Dictionary with security statistics
        """
        try:
            events = self.get_recent_events(limit=1000)

            summary = {
                "total_events": len(events),
                "security_violations": 0,
                "rate_limit_violations": 0,
                "bash_commands": 0,
                "file_operations": 0,
                "sensitive_data_access": 0,
                "violations_by_type": {},
                "most_recent_violation": None
            }

            for event in events:
                event_type = event.get("event_type", "")

                if event_type == "security_violation":
                    summary["security_violations"] += 1
                    violation_type = event.get("violation_type", "unknown")
                    summary["violations_by_type"][violation_type] = \
                        summary["violations_by_type"].get(violation_type, 0) + 1

                    if not summary["most_recent_violation"]:
                        summary["most_recent_violation"] = event

                elif event_type == "rate_limit_exceeded":
                    summary["rate_limit_violations"] += 1

                elif event_type == "bash_command":
                    summary["bash_commands"] += 1

                elif event_type == "file_operation":
                    summary["file_operations"] += 1

                elif event_type == "sensitive_data_access":
                    summary["sensitive_data_access"] += 1

            return summary

        except Exception as e:
            logger.error(f"Failed to generate security summary: {e}")
            return {"error": str(e)}
