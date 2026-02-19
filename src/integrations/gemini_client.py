"""Gemini client via LiteLLM — drop-in for simple text tasks (no tool use).

Used for:
- Intent parsing (cheap, fast, 1M context window)
- Simple/trivial chat responses
- Fallback when Claude hits rate limits (degraded mode, no tools)

NOT used for:
- Tool execution (email, calendar, bash, etc.) — Claude only
- Agent loops with accumulated tool history — Claude only
- Complex reasoning tasks — Claude only
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeminiContentBlock:
    """Mimics anthropic ContentBlock interface."""
    type: str = "text"
    text: str = ""


@dataclass
class GeminiUsage:
    """Mimics anthropic Usage interface."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GeminiResponse:
    """Mimics anthropic.types.Message so ConversationManager needs no changes.

    Callers access: response.content[0].text, response.stop_reason, response.usage
    """
    content: List[GeminiContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: GeminiUsage = field(default_factory=GeminiUsage)


class GeminiClient:
    """LiteLLM-based Gemini client with Anthropic-compatible response format."""

    def __init__(self, api_key: str):
        """Initialize Gemini client.

        Args:
            api_key: Google AI API key (from aistudio.google.com/apikey)
        """
        self.api_key = api_key
        self.enabled = bool(api_key)

        if self.enabled:
            # LiteLLM reads GEMINI_API_KEY from environment
            os.environ["GEMINI_API_KEY"] = api_key
            logger.info("✨ Gemini Flash client initialized (intent + simple chat)")
        else:
            logger.info("Gemini client disabled (no GEMINI_API_KEY)")

    async def create_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs  # absorb unused params (tools, temperature, etc.)
    ) -> GeminiResponse:
        """Send a message to Gemini and return Anthropic-compatible response.

        Only handles plain text messages. Tool-use content blocks are stripped
        (safety guard — Gemini is never used for tool execution paths).

        Args:
            model: LiteLLM model string e.g. "gemini/gemini-2.0-flash"
            messages: Conversation messages (Anthropic or OpenAI format)
            system: System prompt (prepended as system role message)
            max_tokens: Max tokens to generate

        Returns:
            GeminiResponse with .content[0].text, .stop_reason, .usage
        """
        try:
            import litellm
            litellm.suppress_debug_info = True

            # Build OpenAI-format messages for LiteLLM
            litellm_messages = []
            if system:
                litellm_messages.append({"role": "system", "content": system})

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    litellm_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Extract only text blocks — skip tool_use / tool_result blocks
                    text_parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if text_parts:
                        litellm_messages.append({"role": role, "content": " ".join(text_parts)})

            if not litellm_messages:
                return GeminiResponse(
                    content=[GeminiContentBlock(text="")],
                    stop_reason="end_turn"
                )

            response = await litellm.acompletion(
                model=model,
                messages=litellm_messages,
                max_tokens=max_tokens,
                api_key=self.api_key,
            )

            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            usage = response.usage

            logger.info(
                f"Gemini ({model}): {getattr(usage, 'prompt_tokens', 0)} in / "
                f"{getattr(usage, 'completion_tokens', 0)} out"
            )

            return GeminiResponse(
                content=[GeminiContentBlock(type="text", text=text)],
                stop_reason="end_turn" if finish_reason == "stop" else "max_tokens",
                usage=GeminiUsage(
                    input_tokens=getattr(usage, "prompt_tokens", 0),
                    output_tokens=getattr(usage, "completion_tokens", 0),
                ),
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test if Gemini API is reachable.

        Returns:
            True if connection successful
        """
        try:
            response = await self.create_message(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False
