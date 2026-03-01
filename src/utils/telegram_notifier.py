"""Telegram bot for notifications and remote commands."""

import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications and handle commands via Telegram."""

    def __init__(self, bot_token: Optional[str], chat_id: Optional[str]):
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

        if self.enabled:
            # Import telegram only if enabled
            try:
                import telegram
                self.bot = telegram.Bot(token=bot_token)
                logger.info("Telegram bot initialized")
            except ImportError:
                logger.warning("python-telegram-bot not installed. Install with: pip install python-telegram-bot")
                self.enabled = False
        else:
            logger.info("Telegram notifications disabled (no token/chat_id)")

    async def notify(self, message: str, level: str = "info"):
        """Send notification to Telegram.

        Args:
            message: Message to send
            level: Message level (info/success/error/warning)
        """
        if not self.enabled:
            return

        try:
            # Add emoji based on level
            emoji = {
                "info": "ℹ️",
                "success": "✅",
                "error": "❌",
                "warning": "⚠️",
                "progress": "📊"
            }

            formatted_message = f"{emoji.get(level, 'ℹ️')} {message}"

            # Send message
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode="Markdown"
            )

            logger.debug(f"Sent Telegram notification: {level}")

        except Exception as e:
            logger.warning(f"Failed to send Telegram notification (Markdown): {e}")
            try:
                # Fallback to plain text
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode=None
                )
            except Exception as e2:
                logger.error(f"Failed to send Telegram notification (Plain Text): {e2}")

    async def send_progress(
        self,
        phase: str,
        current: int,
        total: int,
        details: str = ""
    ):
        """Send progress update.

        Args:
            phase: Current phase name
            current: Current progress
            total: Total items
            details: Optional details
        """
        if not self.enabled:
            return

        try:
            # Create progress bar
            progress_pct = int((current / total) * 100) if total > 0 else 0
            bar_length = 10
            filled = int((current / total) * bar_length) if total > 0 else 0
            progress_bar = "█" * filled + "░" * (bar_length - filled)

            message = f"""📊 *Progress Update*

Phase: {phase}
Progress: {current}/{total} ({progress_pct}%)
{progress_bar}
"""
            if details:
                message += f"\n{details}"

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Failed to send progress: {e}")

    async def send_build_start(self, total_features: int):
        """Notify build started.

        Args:
            total_features: Total number of features to build
        """
        await self.notify(
            f"🚀 *Starting Autonomous Build*\n\n"
            f"Total features to build: {total_features}\n"
            f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            level="info"
        )

    async def send_build_complete(self, duration_mins: int, features_built: int):
        """Notify build completed.

        Args:
            duration_mins: Build duration in minutes
            features_built: Number of features built
        """
        await self.notify(
            f"✅ *Build Complete!*\n\n"
            f"Features built: {features_built}\n"
            f"Duration: {duration_mins} minutes\n"
            f"System is now operational! 🎉",
            level="success"
        )

    async def send_error(self, error_msg: str, context: str = ""):
        """Send error notification.

        Args:
            error_msg: Error message
            context: Optional context
        """
        message = f"❌ *Error Occurred*\n\n{error_msg}"
        if context:
            message += f"\n\nContext: {context}"

        await self.notify(message, level="error")


class TelegramCommandHandler:
    """Handle incoming commands from Telegram."""

    def __init__(self, bot_token: str, chat_id: str):
        """Initialize command handler.

        Args:
            bot_token: Bot token
            chat_id: Authorized chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.handlers = {}

        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, ContextTypes

            self.Application = Application
            self.CommandHandler = CommandHandler
            self.Update = Update
            self.ContextTypes = ContextTypes
            self.enabled = True

            logger.info("Telegram command handler initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed")
            self.enabled = False

    def register_command(self, command: str, handler: Callable):
        """Register a command handler.

        Args:
            command: Command name (without /)
            handler: Async function to handle command
        """
        self.handlers[command] = handler
        logger.info(f"Registered command: /{command}")

    async def start(self):
        """Start listening for commands."""
        if not self.enabled:
            logger.warning("Command handler not enabled")
            return

        try:
            # Create application
            app = self.Application.builder().token(self.bot_token).build()

            # Register default commands
            self.register_command("start", self._handle_start)
            self.register_command("help", self._handle_help)
            self.register_command("status", self._handle_status)
            self.register_command("report", self._handle_report)

            # Add all registered commands to the application
            for cmd, handler in self.handlers.items():
                app.add_handler(self.CommandHandler(cmd, handler))

            logger.info("Starting Telegram bot polling...")
            await app.run_polling()

        except Exception as e:
            logger.error(f"Error starting command handler: {e}")

    async def _handle_start(self, update, context):
        """Handle /start command."""
        await update.message.reply_text(
            "🤖 *Autonomous Claude Agent*\n\n"
            "I'm your autonomous Non-Human Assistant running 24/7!\n\n"
            "Available commands:\n"
            "/status - Get current status\n"
            "/help - Show help\n"
            "/logs - Get recent logs\n"
            "/build - Start self-building\n"
            "/pause - Pause execution\n"
            "/resume - Resume execution",
            parse_mode="Markdown"
        )

    async def _handle_help(self, update, context):
        """Handle /help command."""
        help_text = """🤖 *Agent Commands*

*Status & Monitoring:*
/status - Current build/execution status
/logs - Recent log entries
/progress - Build progress
/report - Generate an on-demand daily digest report

*Control:*
/build - Start self-building mode
/pause - Pause current execution
/resume - Resume paused execution
/stop - Stop current task

*Info:*
/health - System health check
/features - List built features
/commits - Recent git commits
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _handle_status(self, update, context):
        """Handle /status command."""
        logger.info("Received /status command")
        message = "🟢 *System Status*\n\n"
        message += "Bot is online and ready for tasks.\n"
        message += "Use natural language to chat or ask me to do things."
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def _handle_report(self, update, context):
        """Handle /report command (generates DailyDigest on demand)."""
        logger.info("Received /report command")
        
        # We need to access the DailyDigest instance attached to the agent
        # The easiest way is via the TelegramNotifier's global context or by passing it in
        # We'll try to find it via the orchestrator/agent if available, otherwise suggest waiting for auto-digest
        
        # Try to resolve agent from sys.modules or globals
        try:
            import sys
            main_module = sys.modules.get('__main__')
            if main_module:
                # Need to lookup the daily_digest instance
                # For now, let's create a temporary one just for this report
                from ..self_healing.monitor import SelfHealingMonitor
                from .daily_digest import DailyDigest
                
                # We don't have the exactly right monitor instance, so some stats might be missing
                # but the log parsing and backlog file reading will work fine locally
                temp_digest = DailyDigest(
                    telegram=None,
                    self_healing_monitor=None,
                    log_file="./data/logs/agent.log",
                    data_dir="./data"
                )
                
                await update.message.reply_text("⏳ Generating report from logs...", parse_mode='Markdown')
                report = await temp_digest.generate_report(hours=24)
                await update.message.reply_text(report, parse_mode='Markdown')
                return
        except Exception as e:
            logger.error(f"Error generating on-demand report: {e}")
            
        await update.message.reply_text("❌ Could not generate report. The daily digest will be sent automatically at 9 AM PST.", parse_mode='Markdown')
