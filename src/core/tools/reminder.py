"""Reminder tool — set, list, and cancel reminders with persistent JSON storage."""

import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from .base import BaseTool
from ..timezone import USER_TZ
from ..types import ToolResult

logger = logging.getLogger(__name__)


class ReminderTool(BaseTool):
    """Tool for setting, listing, and cancelling reminders.

    Reminders are stored in data/reminders.json and fired by the
    ReminderScheduler background task via Telegram notifications.
    No external dependencies — always available.
    """

    name = "reminder"
    description = (
        "Set, list, and cancel reminders. Two modes:\n"
        "1. PASSIVE (notify only) — use when no tool action is needed: 'Remind me about John's birthday', "
        "'Remind me to call mom'. Just sends a notification at the specified time.\n"
        "2. ACTIVE (execute action) — use when a task must be performed at a future time: "
        "'Post on LinkedIn at 9 AM', 'Send email tomorrow morning', 'Book restaurant on Mar 28'. "
        "Set action_goal to the full task description. When the reminder fires, Nova will execute "
        "that goal using the right tools automatically — do NOT just set a passive reminder for these."
    )
    parameters = {
        "operation": {
            "type": "string",
            "description": "Operation: 'set_reminder', 'list_reminders', 'cancel_reminder'",
            "enum": ["set_reminder", "list_reminders", "cancel_reminder"]
        },
        "message": {
            "type": "string",
            "description": "Human-readable reminder label shown in notification (for set_reminder)"
        },
        "remind_at": {
            "type": "string",
            "description": "When to fire. Accepts absolute 'YYYY-MM-DD HH:MM' or relative like '30m', '2h', '1d', '90s', '1h30m' (for set_reminder)"
        },
        "action_goal": {
            "type": "string",
            "description": (
                "REQUIRED when the reminder must execute a task (post, send, book, call, etc.). "
                "Write the full goal as you would pass it to nova_task. "
                "Example: 'Post the LinkedIn post from linkedin_post.txt using the linkedin tool'. "
                "Leave empty for passive notify-only reminders."
            )
        },
        "channel": {
            "type": "string",
            "description": "Channel to notify and run the action on when it fires: 'telegram' or 'whatsapp'. Default: 'telegram'.",
            "enum": ["telegram", "whatsapp"]
        },
        "reminder_id": {
            "type": "string",
            "description": "Reminder ID to cancel (for cancel_reminder)"
        }
    }

    def __init__(self, data_dir: str = "./data"):
        """Initialize reminder tool.

        Args:
            data_dir: Directory for persistent storage
        """
        self.data_dir = Path(data_dir)
        self.reminders_file = self.data_dir / "reminders.json"

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Override to make only 'operation' required."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": ["operation"]
            }
        }

    async def execute(
        self,
        operation: str,
        message: Optional[str] = None,
        remind_at: Optional[str] = None,
        action_goal: Optional[str] = None,
        channel: str = "telegram",
        reminder_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute reminder operation."""
        try:
            if operation == "set_reminder":
                return self._set_reminder(message, remind_at, action_goal, channel)
            elif operation == "list_reminders":
                return self._list_reminders()
            elif operation == "cancel_reminder":
                return self._cancel_reminder(reminder_id)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Reminder operation error: {e}", exc_info=True)
            return ToolResult(success=False, error=f"Reminder operation failed: {str(e)}")

    def _set_reminder(
        self,
        message: Optional[str],
        remind_at: Optional[str],
        action_goal: Optional[str] = None,
        channel: str = "telegram",
    ) -> ToolResult:
        """Set a new reminder (passive notify or active action)."""
        if not message:
            return ToolResult(success=False, error="Reminder message is required")
        if not remind_at:
            return ToolResult(success=False, error="remind_at is required. Use 'YYYY-MM-DD HH:MM' or relative like '30m', '2h', '1d'")

        now = datetime.now(USER_TZ)

        # Try relative time first (e.g. "30m", "2h", "1d", "90s", "1h30m")
        remind_dt = self._parse_relative_time(remind_at.strip(), now)

        if not remind_dt:
            # Try absolute datetime
            try:
                remind_dt = datetime.strptime(remind_at.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=USER_TZ)
            except ValueError:
                try:
                    remind_dt = datetime.fromisoformat(remind_at.strip())
                    if remind_dt.tzinfo is None:
                        remind_dt = remind_dt.replace(tzinfo=USER_TZ)
                except ValueError:
                    return ToolResult(
                        success=False,
                        error=f"Invalid time format: '{remind_at}'. Use 'YYYY-MM-DD HH:MM' or relative like '30m', '2h', '1d'"
                    )

        # Reject past reminders
        if remind_dt < now:
            return ToolResult(
                success=False,
                error=f"Cannot set reminder in the past. It's currently {now.strftime('%Y-%m-%d %H:%M')}."
            )

        # Generate unique ID
        rid = uuid.uuid4().hex[:8]

        reminder = {
            "id": rid,
            "message": message,
            "remind_at": remind_dt.isoformat(),
            "created_at": now.isoformat(),
            "status": "pending",
            "channel": channel or "telegram",
        }
        if action_goal and action_goal.strip():
            reminder["action_goal"] = action_goal.strip()

        # Save
        reminders = self._load_reminders()
        reminders.append(reminder)
        self._save_reminders(reminders)

        formatted_time = remind_dt.strftime("%Y-%m-%d %H:%M")
        kind = "action" if reminder.get("action_goal") else "notification"
        logger.info(f"Reminder set ({kind}): {rid} — '{message}' at {formatted_time}")

        return ToolResult(
            success=True,
            output=f"{'Action reminder' if reminder.get('action_goal') else 'Reminder'} set for {formatted_time}: {message} (ID: {rid})",
            metadata={"reminder_id": rid, "remind_at": remind_dt.isoformat(), "has_action": bool(reminder.get("action_goal"))}
        )

    def _list_reminders(self) -> ToolResult:
        """List all pending reminders."""
        reminders = self._load_reminders()

        # Filter pending only
        pending = [r for r in reminders if r.get("status") == "pending"]

        if not pending:
            return ToolResult(success=True, output="No active reminders.")

        # Sort by remind_at
        pending.sort(key=lambda r: r.get("remind_at", ""))

        lines = []
        for i, r in enumerate(pending, 1):
            try:
                dt = datetime.fromisoformat(r["remind_at"])
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, KeyError):
                time_str = r.get("remind_at", "unknown")
            kind = " [ACTION]" if r.get("action_goal") else ""
            lines.append(f"{i}. [{r['id']}]{kind} {time_str} — {r.get('message', 'No message')}")

        return ToolResult(
            success=True,
            output=f"Active reminders ({len(pending)}):\n" + "\n".join(lines)
        )

    def _cancel_reminder(self, reminder_id: Optional[str]) -> ToolResult:
        """Cancel a reminder by ID."""
        if not reminder_id:
            return ToolResult(success=False, error="reminder_id is required")

        reminders = self._load_reminders()

        found = False
        for r in reminders:
            if r.get("id") == reminder_id and r.get("status") == "pending":
                r["status"] = "cancelled"
                r["cancelled_at"] = datetime.now(USER_TZ).isoformat()
                found = True
                break

        if not found:
            return ToolResult(
                success=False,
                error=f"Reminder '{reminder_id}' not found or already fired/cancelled"
            )

        self._save_reminders(reminders)
        logger.info(f"Reminder cancelled: {reminder_id}")

        return ToolResult(success=True, output=f"Reminder {reminder_id} cancelled.")

    def _parse_relative_time(self, value: str, now: datetime) -> Optional[datetime]:
        """Parse relative time strings like '30m', '2h', '1d', '90s', '1h30m'.

        Returns datetime or None if not a relative format.
        """
        # Match patterns like: 30m, 2h, 1d, 90s, 1h30m, 2h15m
        pattern = r'^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$'
        match = re.match(pattern, value.lower().strip())
        if not match:
            return None

        days, hours, minutes, seconds = match.groups()
        if not any([days, hours, minutes, seconds]):
            return None

        delta = timedelta(
            days=int(days or 0),
            hours=int(hours or 0),
            minutes=int(minutes or 0),
            seconds=int(seconds or 0)
        )

        if delta.total_seconds() <= 0:
            return None

        return now + delta

    def _load_reminders(self) -> List[Dict[str, Any]]:
        """Load reminders from JSON file."""
        if not self.reminders_file.exists():
            return []
        try:
            with open(self.reminders_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load reminders file: {e}")
            return []

    def _save_reminders(self, reminders: List[Dict[str, Any]]):
        """Save reminders to JSON file."""
        self.reminders_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reminders_file, 'w') as f:
            json.dump(reminders, f, indent=2, default=str)
