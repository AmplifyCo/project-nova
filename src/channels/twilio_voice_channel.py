"""Twilio Programmable Voice channel.

Handles inbound phone calls via Twilio webhooks.
Uses TwiML to gather speech, passes it to the ConversationManager,
and responds with synthesized speech.
"""

import logging
from typing import Dict, Any, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

logger = logging.getLogger(__name__)

class TwilioVoiceChannel:
    """Twilio Programmable Voice channel adapter."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        phone_number: str,
        conversation_manager
    ):
        """Initialize Twilio Voice channel.

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            phone_number: Twilio Phone Number
            conversation_manager: ConversationManager instance
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.conversation_manager = conversation_manager
        
        self.enabled = bool(account_sid and auth_token and phone_number)
        
        # Default voice settings - User requested Google's voice
        self.voice = "Google.en-US-Journey-F"  # High-quality Google Journey voice
        self.language = "en-US"

        if self.enabled:
            logger.info("✅ Twilio Voice channel initialized")
        else:
            logger.info("Twilio Voice channel disabled (missing credentials)")

    def _generate_twiml(self, text: Optional[str] = None, prompt_for_input: bool = True) -> str:
        """Generate TwiML XML response.
        
        Args:
            text: Text to speak (optional)
            prompt_for_input: Whether to add a Gather verb after speaking
            
        Returns:
            TwiML string
        """
        response = Element('Response')
        
        if text:
            # We use AWS Polly Neural voices built into Twilio for better quality
            say = SubElement(response, 'Say', {'voice': self.voice, 'language': self.language})
            say.text = text
            
        if prompt_for_input:
            # Gather speech input
            # action="/twilio/voice/gather" is where Twilio POSTs the transcribed text
            gather = SubElement(response, 'Gather', {
                'input': 'speech',
                'action': '/twilio/voice/gather',
                'speechTimeout': 'auto',
                'language': self.language
            })
            
            # If no text was provided to say first, say a default greeting inside the gather
            if not text:
                say_gather = SubElement(gather, 'Say', {'voice': self.voice, 'language': self.language})
                say_gather.text = "Hello. How can I help you today?"
                
        # If we didn't prompt for input, and said our piece, end the call
        elif text:
            SubElement(response, 'Hangup')

        # Convert to string with proper XML declaration
        xml_str = tostring(response, encoding='utf-8').decode('utf-8')
        return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{xml_str}"

    async def handle_incoming_call(self, form_data: Dict[str, str]) -> str:
        """Handle initial incoming call webhook (/twilio/voice).
        
        Args:
            form_data: POST form data from Twilio
            
        Returns:
            TwiML string
        """
        if not self.enabled:
            # Fallback if disabled
            response = Element('Response')
            say = SubElement(response, 'Say')
            say.text = "System is currently offline. Please try again later."
            SubElement(response, 'Hangup')
            return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{tostring(response, encoding='utf-8').decode('utf-8')}"
            
        from_number = form_data.get("From", "Unknown")
        call_sid = form_data.get("CallSid", "Unknown")
        
        logger.info(f"📞 Incoming voice call from {from_number} (CallSid: {call_sid})")
        
        # Return initial greeting and start gathering speech
        return self._generate_twiml(text="Hi, I am Nova. How can I help you?", prompt_for_input=True)

    async def handle_gather(self, form_data: Dict[str, str]) -> str:
        """Handle speech recognition result webhook (/twilio/voice/gather).
        
        Args:
            form_data: POST form data from Twilio containing SpeechResult
            
        Returns:
            TwiML string
        """
        if not self.enabled:
            return self._generate_twiml(text="System offline.", prompt_for_input=False)
            
        from_number = form_data.get("From", "Unknown")
        speech_result = form_data.get("SpeechResult", "").strip()
        confidence = form_data.get("Confidence", "0.0")
        
        logger.info(f"🗣️ Speech from {from_number} (confidence {confidence}): {speech_result}")
        
        if not speech_result:
            # Didn't catch that
            return self._generate_twiml(text="I'm sorry, I didn't catch that. Could you please repeat?", prompt_for_input=True)
            
        try:
            # Process via ConversationManager
            ai_response = await self.conversation_manager.process_message(
                message=speech_result,
                channel="voice",
                user_id=from_number,
                enable_periodic_updates=False
            )
            
            # Check if this is a goodbye/end call intent from the AI
            # A simple heuristic: if the AI says goodbye or similar, we might want to hang up.
            # But usually it's better to always prompt for input until the user hangs up.
            is_goodbye = any(phrase in ai_response.lower() for phrase in ["goodbye", "have a great day", "bye for now"])
            
            # Return the AI's response and prompt for more input (unless it's a goodbye)
            return self._generate_twiml(text=ai_response, prompt_for_input=not is_goodbye)
            
        except Exception as e:
            logger.error(f"Voice process error: {e}", exc_info=True)
            return self._generate_twiml(text="I'm sorry, I encountered an error processing your request.", prompt_for_input=True)
