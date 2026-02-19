"""WhatsApp channel adapter using Twilio.

This adapter handles:
1. Receiving inbound messages via Twilio webhook
2. Passing them to ConversationManager
3. Sending responses back via Twilio API
"""

import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WhatsAppChannel:
    """WhatsApp channel adapter using Twilio."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        conversation_manager,
        allowed_numbers: Optional[list] = None
    ):
        """Initialize WhatsApp channel.

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_number: Twilio WhatsApp number (e.g. "whatsapp:+14155238886")
            conversation_manager: ConversationManager instance
            allowed_numbers: List of allowed phone numbers (whitelist)
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.conversation_manager = conversation_manager
        self.allowed_numbers = allowed_numbers or []
        
        self.client = None
        self.enabled = bool(account_sid and auth_token and from_number)

        if self.enabled:
            try:
                from twilio.rest import Client
                self.client = Client(account_sid, auth_token)
                logger.info("✅ WhatsApp channel initialized (Twilio)")
            except ImportError:
                logger.warning("twilio library not installed")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.enabled = False
        else:
            logger.info("WhatsApp channel disabled (missing credentials)")

    async def handle_webhook(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook from Twilio.

        Twilio sends data as form-encoded, but we might receive it parsed 
        depending on how FastAPI handles Request.form().
        
        Args:
            form_data: Parsed form data from Twilio
            
        Returns:
            Response dict (for TwiML or empty)
        """
        if not self.enabled:
            return {"status": "disabled"}

        try:
            # Extract basic info
            # Twilio format: Body, From (whatsapp:+1234567890), To
            body = form_data.get("Body", "")
            from_number = form_data.get("From", "")
            
            # Simple validation
            if not body or not from_number:
                return {"status": "ignored", "reason": "missing_data"}

            # Check whitelist if configured
            # Note: from_number includes "whatsapp:" prefix
            if self.allowed_numbers:
                is_allowed = False
                for allowed in self.allowed_numbers:
                    if allowed in from_number:
                        is_allowed = True
                        break
                
                if not is_allowed:
                    logger.warning(f"🚫 Unauthorized WhatsApp message from {from_number}")
                    return {"status": "ignored", "reason": "unauthorized"}

            logger.info(f"📩 WhatsApp received from {from_number}: {body[:50]}...")

            # Process asynchronously
            asyncio.create_task(self._process_and_respond(body, from_number))

            # Return empty response to valid webhook (Twilio expects 200 OK)
            return {"status": "received"}

        except Exception as e:
            logger.error(f"WhatsApp webhook error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _process_and_respond(self, message: str, user_id: str):
        """Process message and send response.

        Args:
            message: User message
            user_id: User ID (phone number with whatsapp: prefix)
        """
        try:
            # Send "Thinking..." indicator? 
            # WhatsApp doesn't support "typing" actions nicely via API without session costs,
            # so we might skip the "Thinking" status message to keep it clean,
            # or strictly rely on the final response.
            
            # CORE INTELLIGENCE
            response = await self.conversation_manager.process_message(
                message=message,
                channel="whatsapp",
                user_id=user_id,
                # No periodic updates for WhatsApp to avoid spamming multiple messages
                enable_periodic_updates=False 
            )

            # Send final response
            await self.send_message(response, user_id)

        except Exception as e:
            logger.error(f"WhatsApp process error: {e}", exc_info=True)
            # Try to send error message
            await self.send_message(f"❌ Error: {str(e)}", user_id)

    async def send_message(self, text: str, to_number: str):
        """Send message via Twilio.

        Args:
            text: Message text
            to_number: Recipient (must include whatsapp: prefix if using Twilio WhatsApp)
        """
        if not self.enabled or not self.client:
            logger.warning("Attempted to send WhatsApp but channel is disabled")
            return

        try:
            # Run blocking Twilio call in executor
            await asyncio.to_thread(
                self.client.messages.create,
                body=text,
                from_=self.from_number,
                to=to_number
            )
            # logger.info(f"📤 WhatsApp sent to {to_number}") # Redact log effectively?
            logger.info(f"📤 WhatsApp sent to {to_number[:12]}...") 

        except Exception as e:
            logger.error(f"Failed to send WhatsApp: {e}")
