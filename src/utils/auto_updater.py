"""Automatic update system for keeping packages secure and up-to-date."""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .vulnerability_scanner import VulnerabilityScanner, Vulnerability
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class AutoUpdater:
    """Manages automatic updates for Python and system packages."""

    def __init__(
        self,
        bash_tool,
        telegram: Optional[TelegramNotifier] = None,
        config: Dict[str, Any] = None
    ):
        """Initialize auto-updater.

        Args:
            bash_tool: BashTool instance for running commands
            telegram: TelegramNotifier for sending update notifications
            config: Auto-update configuration
        """
        self.bash_tool = bash_tool
        self.telegram = telegram
        self.config = config or {}
        self.scanner = VulnerabilityScanner(bash_tool)

        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.security_only = self.config.get("security_only", True)
        self.auto_restart = self.config.get("auto_restart", True)
        self.notify_telegram = self.config.get("notify_telegram", True)
        self.update_system = self.config.get("packages", {}).get("system", True)
        self.update_python = self.config.get("packages", {}).get("python", True)

        # Backup tracking
        self.backup_dir = Path("data/update_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Update history
        self.update_history: List[Dict[str, Any]] = []
        self.last_update = None

        logger.info(f"AutoUpdater initialized (enabled={self.enabled})")

    async def run_daily_update_check(self):
        """Main entry point - run daily update check and apply updates."""
        if not self.enabled:
            logger.info("Auto-update disabled in config")
            return

        logger.info("🔍 Starting daily update check...")

        try:
            # 1. Scan for vulnerabilities
            await self._notify("🔍 Scanning for vulnerabilities...")
            python_vulns = await self.scanner.scan_python_packages()
            system_updates = await self.scanner.scan_system_packages()

            # 2. Get scan summary
            summary = self.scanner.get_scan_summary()

            # 3. Notify about findings
            await self._send_scan_report(summary, python_vulns, system_updates)

            # 4. Apply updates if needed
            updates_applied = False

            if self.update_python and python_vulns:
                updates_applied |= await self._update_python_packages(python_vulns)

            if self.update_system and system_updates:
                updates_applied |= await self._update_system_packages(system_updates)

            # 5. Restart if needed and updates were applied
            if updates_applied and self.auto_restart:
                await self._restart_agent()

            # 6. Final notification
            if updates_applied:
                await self._notify("✅ Auto-update completed successfully", "success")
            else:
                await self._notify("✅ All packages up-to-date", "info")

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error during auto-update: {e}", exc_info=True)
            await self._notify(f"❌ Auto-update failed: {str(e)}", "error")

    async def _update_python_packages(self, vulnerabilities: List[Vulnerability]) -> bool:
        """Update vulnerable Python packages.

        Args:
            vulnerabilities: List of vulnerabilities to fix

        Returns:
            True if updates were applied
        """
        if not vulnerabilities:
            return False

        logger.info(f"Updating {len(vulnerabilities)} vulnerable Python packages...")

        # Filter by severity if security_only
        to_update = vulnerabilities
        if self.security_only:
            to_update = [v for v in vulnerabilities if v.severity in ["critical", "high"]]

        if not to_update:
            logger.info("No critical/high vulnerabilities to fix")
            return False

        # Create backup of current requirements
        await self._backup_requirements()

        updated_packages = []
        failed_packages = []

        for vuln in to_update:
            try:
                # Determine version to install
                if vuln.fixed_version:
                    install_spec = f"{vuln.package}=={vuln.fixed_version}"
                else:
                    install_spec = f"{vuln.package} --upgrade"

                logger.info(f"Updating {vuln.package}: {vuln.installed_version} -> {vuln.fixed_version or 'latest'}")

                result = await self.bash_tool.execute(
                    f"pip install --upgrade {install_spec}",
                    timeout=180
                )

                if result.success:
                    updated_packages.append(vuln.package)
                    logger.info(f"✅ Updated {vuln.package}")
                else:
                    failed_packages.append(vuln.package)
                    logger.error(f"❌ Failed to update {vuln.package}: {result.error}")

            except Exception as e:
                logger.error(f"Error updating {vuln.package}: {e}")
                failed_packages.append(vuln.package)

        # Record update
        self._record_update({
            "type": "python",
            "packages_updated": updated_packages,
            "packages_failed": failed_packages,
            "timestamp": datetime.now().isoformat()
        })

        # Notify
        if updated_packages:
            await self._notify(
                f"📦 Updated {len(updated_packages)} Python packages:\n" +
                "\n".join(f"  • {pkg}" for pkg in updated_packages[:10]),
                "success"
            )

        if failed_packages:
            await self._notify(
                f"⚠️ Failed to update {len(failed_packages)} packages:\n" +
                "\n".join(f"  • {pkg}" for pkg in failed_packages),
                "warning"
            )

        return len(updated_packages) > 0

    async def _update_system_packages(self, updates: List[Dict[str, str]]) -> bool:
        """Update system packages.

        Args:
            updates: List of available system updates

        Returns:
            True if updates were applied
        """
        if not updates:
            return False

        logger.info(f"Applying {len(updates)} system security updates...")

        try:
            # Apply security updates only
            result = await self.bash_tool.execute(
                "sudo yum update --security -y",
                timeout=600
            )

            if result.success:
                await self._notify(
                    f"🔒 Applied {len(updates)} system security updates",
                    "success"
                )

                self._record_update({
                    "type": "system",
                    "updates_count": len(updates),
                    "timestamp": datetime.now().isoformat()
                })

                return True
            else:
                logger.error(f"Failed to apply system updates: {result.error}")
                await self._notify(
                    f"❌ System update failed: {result.error}",
                    "error"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating system packages: {e}")
            await self._notify(f"❌ System update error: {str(e)}", "error")
            return False

    async def _backup_requirements(self):
        """Backup current pip requirements before updating."""
        try:
            result = await self.bash_tool.execute("pip freeze")

            if result.success:
                backup_file = self.backup_dir / f"requirements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                backup_file.write_text(result.output)
                logger.info(f"Backed up requirements to {backup_file}")

        except Exception as e:
            logger.error(f"Error backing up requirements: {e}")

    async def _restart_agent(self):
        """Restart the agent service after updates."""
        logger.info("Restarting agent after updates...")

        await self._notify("🔄 Restarting agent to apply updates...", "info")

        try:
            # Give time for notification to be sent
            await asyncio.sleep(2)

            # Restart via systemd
            result = await self.bash_tool.execute(
                "sudo systemctl restart claude-agent",
                timeout=30
            )

            if not result.success:
                logger.error(f"Failed to restart agent: {result.error}")

        except Exception as e:
            logger.error(f"Error restarting agent: {e}")

    async def _send_scan_report(
        self,
        summary: Dict[str, Any],
        python_vulns: List[Vulnerability],
        system_updates: List[Dict[str, str]]
    ):
        """Send vulnerability scan report via Telegram.

        Args:
            summary: Scan summary
            python_vulns: Python vulnerabilities found
            system_updates: System updates available
        """
        if not self.notify_telegram or not self.telegram or not self.telegram.enabled:
            return

        # Build report message
        message = "🔍 *Security Scan Report*\n\n"

        # Python vulnerabilities
        if python_vulns:
            message += f"*Python Packages:*\n"
            message += f"  • Total vulnerabilities: {summary['total_vulnerabilities']}\n"
            message += f"  • Critical: {summary['severity_breakdown']['critical']}\n"
            message += f"  • High: {summary['severity_breakdown']['high']}\n"
            message += f"  • Medium: {summary['severity_breakdown']['medium']}\n\n"

            if summary['critical_packages']:
                message += "*Critical packages:*\n"
                for pkg in summary['critical_packages'][:5]:
                    message += f"  • {pkg}\n"
                message += "\n"
        else:
            message += "*Python Packages:* ✅ No vulnerabilities\n\n"

        # System updates
        if system_updates:
            message += f"*System Updates:* {len(system_updates)} security updates available\n"
        else:
            message += "*System Updates:* ✅ Up to date\n"

        await self._notify(message, "info")

    def _record_update(self, update_info: Dict[str, Any]):
        """Record update in history.

        Args:
            update_info: Update information to record
        """
        self.update_history.append(update_info)

        # Keep only last 50 updates
        if len(self.update_history) > 50:
            self.update_history = self.update_history[-50:]

        # Save to file
        try:
            history_file = self.backup_dir / "update_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.update_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update history: {e}")

    async def _notify(self, message: str, level: str = "info"):
        """Send notification via Telegram.

        Args:
            message: Message to send
            level: Notification level
        """
        if self.notify_telegram and self.telegram and self.telegram.enabled:
            try:
                await self.telegram.notify(message, level=level)
            except Exception as e:
                logger.error(f"Error sending Telegram notification: {e}")

    async def start_background_task(self):
        """Start background task that runs daily updates."""
        logger.info("Starting auto-update background task...")

        while True:
            try:
                # Run daily update check
                await self.run_daily_update_check()

                # Wait 24 hours
                logger.info("Next auto-update check in 24 hours")
                await asyncio.sleep(86400)  # 24 hours

            except asyncio.CancelledError:
                logger.info("Auto-update task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto-update background task: {e}")
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)

    def get_status(self) -> Dict[str, Any]:
        """Get auto-update status.

        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_scan": self.scanner.last_scan.isoformat() if self.scanner.last_scan else None,
            "vulnerabilities_found": len(self.scanner.vulnerabilities),
            "critical_vulnerabilities": len(self.scanner.get_critical_vulnerabilities()),
            "update_history_count": len(self.update_history),
            "config": {
                "security_only": self.security_only,
                "auto_restart": self.auto_restart,
                "update_system": self.update_system,
                "update_python": self.update_python
            }
        }
