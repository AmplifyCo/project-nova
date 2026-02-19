"""WhatsApp Tool for sending messages via Twilio."""

from typing import Dict, Any, Optional
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class WhatsAppTool(BaseTool):
    """Tool for sending WhatsApp messages via Twilio."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        """Initialize WhatsApp tool.

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_number: Twilio WhatsApp number (e.g. "whatsapp:+14155238886")
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.client = None
        self.enabled = bool(account_sid and auth_token and from_number)

        if self.enabled:
            try:
                from twilio.rest import Client
                self.client = Client(account_sid, auth_token)
            except ImportError:
                logger.warning("twilio library not installed")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.enabled = False

    @property
    def name(self) -> str:
        return "whatsapp"

    @property
    def description(self) -> str:
        return "Send WhatsApp messages to known contacts."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient phone number (e.g. +1234567890). Must include country code."
                },
                "body": {
                    "type": "string",
                    "description": "Message content to send"
                }
            },
            "required": ["to", "body"]
        }

    async def execute(self, to: str, body: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool.

        Args:
            to: Recipient phone number
            body: Message content

        Returns:
            Result dictionary
        """
        if not self.enabled:
            return {"success": False, "error": "WhatsApp tool disabled (missing credentials)"}

        try:
            # Twilio requires "whatsapp:" prefix for WhatsApp messages
            # If "to" doesn't have it, add it.
            # (Note: PII system might return raw number like +1234567890)
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to}"
            
            # Same for "from" if not configured
            from_ = self.from_number
            if not from_.startswith("whatsapp:"):
                # Usually config has it, but just in case
                from_ = f"whatsapp:{from_}"

            # Run blocking call in executor
            import asyncio
            message = await asyncio.to_thread(
                self.client.messages.create,
                body=body,
                from_=from_,
                to=to
            )

            logger.info(f"📤 WhatsApp sent to {to} (SID: {message.sid})")
            return {
                "success": True, 
                "output": f"Message sent successfully to {to}", 
                "message_sid": message.sid
            }

        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return {"success": False, "error": str(e)}
