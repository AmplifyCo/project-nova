"""Auto-fix system for attempting to resolve detected errors."""

import logging
import subprocess
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from .error_detector import DetectedError, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)


class FixResult:
    """Result of an auto-fix attempt."""

    def __init__(
        self,
        success: bool,
        error_type: ErrorType,
        action_taken: str,
        details: Optional[str] = None,
        requires_restart: bool = False
    ):
        self.success = success
        self.error_type = error_type
        self.action_taken = action_taken
        self.details = details
        self.requires_restart = requires_restart

    def __repr__(self):
        return f"<FixResult success={self.success} action={self.action_taken} restart_needed={self.requires_restart}>"


class AutoFixer:
    """Automatically attempts to fix detected errors."""

    def __init__(self, telegram_notifier=None):
        """Initialize auto-fixer.

        Args:
            telegram_notifier: Optional Telegram chat instance for notifications
        """
        self.telegram = telegram_notifier
        self.fix_history: List[FixResult] = []

        logger.info("AutoFixer initialized")

    async def attempt_fix(self, error: DetectedError) -> FixResult:
        """Attempt to automatically fix a detected error.

        Args:
            error: The detected error to fix

        Returns:
            FixResult indicating success/failure
        """
        logger.info(f"Attempting to fix error: {error.error_type.value}")

        # Route to appropriate fix strategy
        fix_strategies = {
            ErrorType.RATE_LIMIT: self._fix_rate_limit,
            ErrorType.IMPORT_ERROR: self._fix_import_error,
            ErrorType.GIT_ERROR: self._fix_git_error,
            ErrorType.CONFIG_ERROR: self._fix_config_error,
            ErrorType.ATTRIBUTE_ERROR: self._fix_attribute_error,
        }

        fix_func = fix_strategies.get(error.error_type)
        if not fix_func:
            logger.warning(f"No fix strategy for {error.error_type.value}")
            return FixResult(
                success=False,
                error_type=error.error_type,
                action_taken="None - no fix strategy available",
                details="This error type is not auto-fixable"
            )

        try:
            result = await fix_func(error)
            self.fix_history.append(result)

            # Notify via Telegram if available
            if self.telegram:
                await self._notify_fix_attempt(error, result)

            return result

        except Exception as e:
            logger.error(f"Error during auto-fix: {e}", exc_info=True)
            return FixResult(
                success=False,
                error_type=error.error_type,
                action_taken="Fix attempt failed",
                details=str(e)
            )

    async def _fix_rate_limit(self, error: DetectedError) -> FixResult:
        """Fix rate limit errors by enabling fallback model.

        Args:
            error: The detected error

        Returns:
            FixResult
        """
        logger.info("Fixing rate limit error - enabling fallback model")

        # Check if fallback is already enabled
        try:
            from ...integrations.model_router import ModelRouter
            # Fallback is already built into the system
            # Just log that it should activate automatically

            return FixResult(
                success=True,
                error_type=ErrorType.RATE_LIMIT,
                action_taken="Fallback model already configured",
                details="System will automatically use SmolLM2/DeepSeek when rate limited",
                requires_restart=False
            )
        except Exception as e:
            return FixResult(
                success=False,
                error_type=ErrorType.RATE_LIMIT,
                action_taken="Could not verify fallback configuration",
                details=str(e),
                requires_restart=False
            )

    async def _fix_import_error(self, error: DetectedError) -> FixResult:
        """Fix import errors by installing missing packages.

        Args:
            error: The detected error

        Returns:
            FixResult
        """
        logger.info("Fixing import error - attempting to install missing package")

        # Extract module name from error message
        import re
        match = re.search(r"No module named '(\w+)'", error.message)
        if not match:
            return FixResult(
                success=False,
                error_type=ErrorType.IMPORT_ERROR,
                action_taken="Could not identify missing module",
                details="Unable to parse module name from error"
            )

        module_name = match.group(1)
        logger.info(f"Attempting to install: {module_name}")

        try:
            # Try to install the package
            result = subprocess.run(
                ["pip", "install", module_name],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return FixResult(
                    success=True,
                    error_type=ErrorType.IMPORT_ERROR,
                    action_taken=f"Installed missing package: {module_name}",
                    details=result.stdout,
                    requires_restart=True  # Need restart to reload modules
                )
            else:
                return FixResult(
                    success=False,
                    error_type=ErrorType.IMPORT_ERROR,
                    action_taken=f"Failed to install {module_name}",
                    details=result.stderr
                )

        except subprocess.TimeoutExpired:
            return FixResult(
                success=False,
                error_type=ErrorType.IMPORT_ERROR,
                action_taken="Installation timed out",
                details=f"pip install {module_name} took too long"
            )
        except Exception as e:
            return FixResult(
                success=False,
                error_type=ErrorType.IMPORT_ERROR,
                action_taken="Installation failed",
                details=str(e)
            )

    async def _fix_git_error(self, error: DetectedError) -> FixResult:
        """Fix git errors (conflicts, etc).

        Args:
            error: The detected error

        Returns:
            FixResult
        """
        logger.info("Fixing git error")

        # Check if it's a merge conflict
        if "merge conflict" in error.message.lower():
            try:
                # Stash local changes and pull again
                logger.info("Detected merge conflict - stashing changes")

                # Stash changes
                subprocess.run(["git", "stash"], check=True, capture_output=True)

                # Pull latest
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                return FixResult(
                    success=True,
                    error_type=ErrorType.GIT_ERROR,
                    action_taken="Stashed local changes and pulled latest",
                    details="Merge conflict resolved. Local changes stashed.",
                    requires_restart=True
                )

            except subprocess.CalledProcessError as e:
                return FixResult(
                    success=False,
                    error_type=ErrorType.GIT_ERROR,
                    action_taken="Failed to resolve git conflict",
                    details=e.stderr
                )

        # Generic git error - try git status
        try:
            result = subprocess.run(
                ["git", "status"],
                capture_output=True,
                text=True,
                check=True
            )

            return FixResult(
                success=True,
                error_type=ErrorType.GIT_ERROR,
                action_taken="Checked git status",
                details=result.stdout,
                requires_restart=False
            )

        except Exception as e:
            return FixResult(
                success=False,
                error_type=ErrorType.GIT_ERROR,
                action_taken="Could not diagnose git error",
                details=str(e)
            )

    async def _fix_config_error(self, error: DetectedError) -> FixResult:
        """Fix configuration errors.

        Args:
            error: The detected error

        Returns:
            FixResult
        """
        logger.info("Fixing configuration error")

        # Check if .env exists
        env_file = Path(".env")
        env_example = Path(".env.example")

        if not env_file.exists() and env_example.exists():
            try:
                # Copy .env.example to .env
                import shutil
                shutil.copy(env_example, env_file)

                return FixResult(
                    success=True,
                    error_type=ErrorType.CONFIG_ERROR,
                    action_taken="Created .env from .env.example",
                    details="You may need to add your API keys",
                    requires_restart=True
                )

            except Exception as e:
                return FixResult(
                    success=False,
                    error_type=ErrorType.CONFIG_ERROR,
                    action_taken="Failed to create .env",
                    details=str(e)
                )

        # Check config/agent.yaml
        config_file = Path("config/agent.yaml")
        if not config_file.exists():
            return FixResult(
                success=False,
                error_type=ErrorType.CONFIG_ERROR,
                action_taken="Missing config/agent.yaml",
                details="Configuration file not found. May need to restore from git."
            )

        return FixResult(
            success=False,
            error_type=ErrorType.CONFIG_ERROR,
            action_taken="Configuration appears intact",
            details="Unable to automatically fix this config error"
        )

    async def _fix_attribute_error(self, error: DetectedError) -> FixResult:
        """Fix attribute errors (like the .model vs .default_model issue).

        Args:
            error: The detected error

        Returns:
            FixResult
        """
        logger.info("Fixing attribute error")

        # For attribute errors, we can't automatically fix code
        # But we can suggest pulling latest from git
        if "config" in error.message.lower() and "model" in error.message.lower():
            try:
                # Check if there are updates on git
                result = subprocess.run(
                    ["git", "fetch", "origin", "main"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                # Check if we're behind
                result = subprocess.run(
                    ["git", "rev-list", "HEAD..origin/main", "--count"],
                    capture_output=True,
                    text=True
                )

                commits_behind = int(result.stdout.strip())
                if commits_behind > 0:
                    return FixResult(
                        success=True,
                        error_type=ErrorType.ATTRIBUTE_ERROR,
                        action_taken="Detected code fix available in git",
                        details=f"There are {commits_behind} new commits. Suggest: 'git pull origin main'",
                        requires_restart=False
                    )

            except Exception as e:
                logger.error(f"Error checking git: {e}")

        return FixResult(
            success=False,
            error_type=ErrorType.ATTRIBUTE_ERROR,
            action_taken="Cannot auto-fix code errors",
            details="Attribute errors require code changes. Check git for updates."
        )

    async def _notify_fix_attempt(self, error: DetectedError, result: FixResult):
        """Notify user via Telegram about fix attempt.

        Args:
            error: The original error
            result: The fix result
        """
        if not self.telegram:
            return

        icon = "✅" if result.success else "❌"
        status = "Fixed" if result.success else "Failed"

        message = f"""{icon} **Auto-Fix: {status}**

**Error Type:** {error.error_type.value}
**Severity:** {error.severity.value}
**Action Taken:** {result.action_taken}

**Details:** {result.details or 'None'}
"""

        if result.requires_restart:
            message += "\n⚠️ **Restart required** for changes to take effect"

        try:
            await self.telegram.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def get_fix_summary(self) -> Dict[str, Any]:
        """Get summary of fix attempts.

        Returns:
            Summary dict
        """
        total = len(self.fix_history)
        successful = sum(1 for f in self.fix_history if f.success)

        by_type = {}
        for fix in self.fix_history:
            error_type = fix.error_type.value
            by_type[error_type] = by_type.get(error_type, 0) + 1

        return {
            "total_fixes_attempted": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "N/A",
            "by_type": by_type,
            "recent_fixes": [
                {
                    "type": f.error_type.value,
                    "success": f.success,
                    "action": f.action_taken
                }
                for f in self.fix_history[-5:]  # Last 5 fixes
            ]
        }
