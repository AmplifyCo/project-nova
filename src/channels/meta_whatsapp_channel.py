"""Meta Cloud API WhatsApp channel adapter.

Uses the Meta (Facebook) Cloud API directly — the only way to send
read receipts (blue ticks) to the user.

Flow for each incoming message:
  1. Immediately mark as read  → blue ticks appear on sender's phone
  2. Fire background task      → process with ConversationManager
  3. Reply via REST API        → message delivered asynchronously

Webhook setup in Meta Developer Console:
  - Webhook URL:   https://<your-domain>/whatsapp/webhook
  - Verify token:  value of WHATSAPP_VERIFY_TOKEN env var
  - Subscribe to:  messages
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

_META_API_BASE = "https://graph.facebook.com/v21.0"


class MetaWhatsAppChannel:
    """Meta Cloud API WhatsApp channel adapter."""

    def __init__(
        self,
        api_token: str,
        phone_number_id: str,
        verify_token: str,
        conversation_manager,
        allowed_numbers: Optional[List[str]] = None,
    ):
        self.api_token = api_token
        self.phone_number_id = phone_number_id
        self.verify_token = verify_token
        self.conversation_manager = conversation_manager
        self.allowed_numbers = [n.lstrip("+") for n in (allowed_numbers or [])]

        self.enabled = bool(api_token and phone_number_id and verify_token)

        if self.enabled:
            logger.info(f"✅ Meta WhatsApp channel initialized (phone_number_id={phone_number_id})")
        else:
            logger.info("Meta WhatsApp channel disabled (missing credentials)")

    # ── Webhook verification ──────────────────────────────────────────────────

    def handle_verification(self, params: Dict[str, str]) -> Optional[str]:
        """Handle Meta webhook verification GET request.

        Returns the hub.challenge string on success, None on failure.
        """
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")

        if mode == "subscribe" and token == self.verify_token:
            logger.info("Meta WhatsApp webhook verified successfully")
            return challenge

        logger.warning(f"Meta webhook verification failed (token mismatch or wrong mode='{mode}')")
        return None

    # ── Incoming message ──────────────────────────────────────────────────────

    async def handle_webhook(self, body: Dict[str, Any]) -> None:
        """Handle incoming Meta Cloud API webhook POST.

        Sends a read receipt immediately, then processes the message in the
        background so the HTTP response returns to Meta without waiting for the AI.
        """
        if not self.enabled:
            return

        try:
            entry = body.get("entry", [])
            if not entry:
                return

            for e in entry:
                for change in e.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    for msg in messages:
                        msg_id = msg.get("id", "")
                        from_number = msg.get("from", "")  # plain digits, no +

                        # Only handle text messages for now
                        msg_type = msg.get("type", "")
                        if msg_type != "text":
                            logger.debug(f"Ignoring non-text message type='{msg_type}' from {from_number}")
                            continue

                        text_body = msg.get("text", {}).get("body", "").strip()
                        if not text_body:
                            continue

                        # Access control
                        if not self._is_allowed(from_number):
                            logger.warning(f"Meta WhatsApp: message from unauthorized number {from_number}")
                            continue

                        logger.info(f"💬 Meta WhatsApp message from {from_number}: {text_body[:60]}...")

                        # 1. Send read receipt immediately → blue ticks appear
                        asyncio.create_task(self._send_read_receipt(msg_id))

                        # 2. Process + reply in background → no blocking
                        asyncio.create_task(self._process_and_reply(from_number, text_body))

        except Exception as e:
            logger.error(f"Error handling Meta WhatsApp webhook: {e}", exc_info=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_allowed(self, phone_number: str) -> bool:
        if not self.allowed_numbers:
            return True
        clean = phone_number.lstrip("+")
        return clean in self.allowed_numbers

    async def _send_read_receipt(self, message_id: str) -> None:
        """Mark a message as read so the sender sees blue ticks."""
        url = f"{_META_API_BASE}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    logger.debug(f"Read receipt sent for {message_id}")
                else:
                    logger.warning(f"Read receipt failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Failed to send read receipt: {e}")

    async def _process_and_reply(self, from_number: str, message_body: str) -> None:
        """Process the message with ConversationManager and send the reply."""
        try:
            owner_name = os.getenv("OWNER_NAME", "User")
            clean_allowed = [n.lstrip("+") for n in self.allowed_numbers]
            clean_number = from_number.lstrip("+")

            user_id = f"{owner_name} (Principal)" if clean_number in clean_allowed else clean_number

            ai_response = await self.conversation_manager.process_message(
                message=message_body,
                channel="whatsapp",
                user_id=user_id,
            )

            await self.send_message(to=from_number, body=ai_response)

        except Exception as e:
            logger.error(f"Error in Meta WhatsApp _process_and_reply: {e}", exc_info=True)
            await self.send_message(
                to=from_number,
                body="Sorry, I encountered an error processing your request."
            )

    async def send_message(self, to: str, body: str) -> bool:
        """Send a WhatsApp text message via Meta Cloud API.

        Args:
            to: Recipient phone number (digits only, with or without leading +)
            body: Message text
        """
        if not self.enabled:
            logger.error("Cannot send message: Meta WhatsApp channel disabled")
            return False

        to_clean = to.lstrip("+")
        url = f"{_META_API_BASE}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_clean,
            "type": "text",
            "text": {"body": body},
        }
        headers = {"Authorization": f"Bearer {self.api_token}"}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    logger.info(f"✅ Meta WhatsApp message sent to {to_clean}")
                    return True
                else:
                    logger.error(f"Meta WhatsApp send failed ({resp.status_code}): {resp.text[:300]}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Meta WhatsApp message: {e}")
            return False
