"""Intelligent model router for optimal model selection.

PRIORITY: Clarity and quality over cost savings.
Only use lightweight models for simple, unambiguous tasks.

FALLBACK: If Claude API fails, automatically fall back to SmolLM2 local model.
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class APIFailureMode(Enum):
    """API failure handling modes."""
    FAIL_FAST = "fail_fast"           # Raise error immediately
    FALLBACK_LOCAL = "fallback_local"  # Fall back to local model
    RETRY_THEN_FALLBACK = "retry_then_fallback"  # Retry, then fall back


class ModelTier(Enum):
    """Model tiers with clear use cases."""
    LOCAL = "local"          # Ultra-simple, predefined responses only
    HAIKU = "haiku"          # Simple intent parsing, straightforward queries
    SONNET = "sonnet"        # General conversation, moderate complexity (DEFAULT)
    OPUS = "opus"            # Complex planning, feature building, architect


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"      # Predefined responses (status, health)
    SIMPLE = "simple"        # Clear intent, straightforward (git pull, logs)
    MODERATE = "moderate"    # Requires understanding, conversation
    COMPLEX = "complex"      # Planning, building, orchestration


class ModelRouter:
    """Intelligent router for selecting the right model.

    Philosophy:
    - Prioritize CLARITY and QUALITY over cost
    - Use lightweight models only when safe and appropriate
    - When in doubt, escalate to better model
    - Default to Sonnet for general tasks
    """

    def __init__(self, config):
        """Initialize router with configuration.

        Args:
            config: AgentConfig with model settings
        """
        self.config = config

        # Model tier mapping
        self.models = {
            ModelTier.OPUS: config.default_model,
            ModelTier.SONNET: config.subagent_model,
            ModelTier.HAIKU: config.chat_model,
            ModelTier.LOCAL: "local"  # Placeholder
        }

        # Gemini config (optional second provider via LiteLLM)
        self.gemini_model = getattr(config, "gemini_model", "gemini/gemini-2.0-flash")
        self.gemini_enabled = getattr(config, "gemini_enabled", False)

        if self.gemini_enabled:
            logger.info(f"✨ Gemini Flash enabled for intent/simple tasks ({self.gemini_model})")
        logger.info(f"Initialized ModelRouter - Default: {config.subagent_model} (Sonnet)")

    def select_model_for_task(
        self,
        task: str,
        intent: Optional[str] = None,
        confidence: float = 0.0,
        user_context: Optional[str] = None
    ) -> str:
        """Select appropriate model for a task.

        Args:
            task: Task description or user message
            intent: Parsed intent (if available)
            confidence: Intent confidence (0.0-1.0)
            user_context: Additional context

        Returns:
            Model name to use
        """
        complexity = self._assess_complexity(task, intent, confidence)
        tier = self._map_complexity_to_tier(complexity)

        model = self.models[tier]
        logger.info(f"Selected {tier.value} model ({model}) for {complexity.value} task")

        return model

    def _assess_complexity(
        self,
        task: str,
        intent: Optional[str] = None,
        confidence: float = 0.0
    ) -> TaskComplexity:
        """Assess task complexity.

        Args:
            task: Task description
            intent: Parsed intent
            confidence: Intent confidence

        Returns:
            TaskComplexity level
        """
        task_lower = task.lower()

        # TRIVIAL: Predefined, deterministic responses
        trivial_intents = ["status", "health"]
        if intent in trivial_intents and confidence > 0.8:
            return TaskComplexity.TRIVIAL

        # SIMPLE: Clear, straightforward actions
        simple_intents = ["git_pull", "git_update", "logs", "restart"]
        simple_keywords = ["pull", "update", "restart", "status", "health", "logs"]

        if intent in simple_intents and confidence > 0.7:
            return TaskComplexity.SIMPLE

        if any(kw in task_lower for kw in simple_keywords) and len(task.split()) < 8:
            return TaskComplexity.SIMPLE

        # COMPLEX: Building, planning, unclear intent
        complex_intents = ["build_feature", "unknown"]
        complex_keywords = ["build", "create", "implement", "design", "plan", "develop"]

        if intent == "build_feature":
            return TaskComplexity.COMPLEX

        if intent == "unknown" or confidence < 0.6:
            # Low confidence = escalate to complex
            return TaskComplexity.COMPLEX

        if any(kw in task_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX

        # MODERATE: Everything else (general conversation, questions)
        # This is the DEFAULT - when in doubt, use Sonnet
        return TaskComplexity.MODERATE

    def _map_complexity_to_tier(self, complexity: TaskComplexity) -> ModelTier:
        """Map complexity to model tier.

        IMPORTANT: Prioritizes quality. Only uses lightweight models
        when absolutely safe.

        Args:
            complexity: Task complexity

        Returns:
            ModelTier to use
        """
        mapping = {
            TaskComplexity.TRIVIAL: ModelTier.HAIKU,    # Safe - predefined responses
            TaskComplexity.SIMPLE: ModelTier.HAIKU,     # Safe - clear intent
            TaskComplexity.MODERATE: ModelTier.SONNET,  # DEFAULT - quality matters
            TaskComplexity.COMPLEX: ModelTier.OPUS      # Architect for complex tasks
        }

        tier = mapping.get(complexity, ModelTier.SONNET)  # Default to Sonnet

        # Override for local model if enabled and appropriate
        if (self.config.local_model_enabled and
            complexity == TaskComplexity.TRIVIAL and
            self._is_local_model_safe()):
            return ModelTier.LOCAL

        return tier

    def _is_local_model_safe(self) -> bool:
        """Check if it's safe to use local model.

        Local models should ONLY be used for ultra-simple tasks
        where quality doesn't matter (e.g., "OK", "Running").

        Returns:
            True if safe to use local model
        """
        # Only use local for tasks in the allowed list
        allowed_for_local = self.config.local_model_for.split(",")

        # Currently, we're conservative - only allow for trivial tasks
        return "trivial" in allowed_for_local or "simple" in allowed_for_local

    def select_model_for_intent_parsing(self) -> str:
        """Select model for intent parsing.

        Intent parsing is straightforward - Haiku is perfect.

        Returns:
            Model name for intent parsing
        """
        return self.config.intent_model

    def select_model_for_chat(self, message_length: int = 0) -> str:
        """Select model for conversational chat.

        IMPORTANT: Chat requires understanding context and nuance.
        Default to Sonnet for quality unless message is trivial.

        Args:
            message_length: Length of message

        Returns:
            Model name for chat
        """
        # Very short messages might be simple greetings
        if message_length < 20:
            simple_greetings = ["hi", "hello", "hey", "thanks", "ok", "yes", "no"]
            # Even then, use Haiku (not local) for basic quality
            return self.config.chat_model

        # For actual conversation, use Sonnet for quality
        return self.config.subagent_model

    def select_model_for_architect(self) -> str:
        """Select model for architect/orchestrator role.

        Always use Opus for architectural decisions.

        Returns:
            Model name for architect
        """
        return self.config.default_model

    def get_intent_provider(self) -> tuple:
        """Return (provider, model) for intent parsing.

        Gemini Flash is preferred — it's faster, cheaper, and has 1M token context.
        Falls back to Claude Haiku if Gemini not configured.

        Returns:
            Tuple of ("gemini"|"claude", model_name)
        """
        if self.gemini_enabled:
            return "gemini", self.gemini_model
        return "claude", self.config.intent_model

    def get_fallback_provider(self) -> tuple:
        """Return (provider, model) for when Claude hits rate limits.

        Gemini Flash is a much better fallback than local SmolLM2 — it's
        a real LLM with full reasoning capability.

        Returns:
            Tuple of ("gemini"|"local"|None, model_name|None)
        """
        if self.gemini_enabled:
            return "gemini", self.gemini_model
        if self.config.local_model_enabled:
            return "local", self.config.local_model_name
        return None, None

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model: Model name

        Returns:
            Model info dict
        """
        info = {
            "opus-4-6": {
                "tier": "architect",
                "cost": "high",
                "speed": "medium",
                "quality": "highest",
                "use_for": "complex planning, orchestration"
            },
            "sonnet-4-5": {
                "tier": "worker",
                "cost": "medium",
                "speed": "fast",
                "quality": "high",
                "use_for": "implementation, conversation"
            },
            "haiku-4-5": {
                "tier": "assistant",
                "cost": "low",
                "speed": "very fast",
                "quality": "good",
                "use_for": "simple tasks, intent parsing"
            },
            "local": {
                "tier": "basic",
                "cost": "free",
                "speed": "varies",
                "quality": "basic",
                "use_for": "ultra-simple predefined responses"
            }
        }

        # Extract model key
        for key in info:
            if key in model:
                return info[key]

        return {"tier": "unknown", "cost": "unknown", "quality": "unknown"}

    def get_fallback_model(self) -> Optional[str]:
        """Get fallback model when Claude API fails.

        Returns:
            Local model name if enabled, None otherwise
        """
        if self.config.local_model_enabled:
            logger.info(f"Using fallback model: {self.config.local_model_name}")
            return "local"  # Special identifier for local model
        return None

    def should_use_fallback(self, error: Exception) -> bool:
        """Determine if we should fall back to alternate provider.

        Args:
            error: The exception that occurred

        Returns:
            True if should use fallback
        """
        error_str = str(error).lower()

        # Rate limit errors (429)
        if "429" in error_str or "resource exhausted" in error_str:
            logger.warning(f"Rate limit error detected: {error}")
            return True

        # Server errors (5xx)
        if any(x in error_str for x in ["500", "502", "503", "overloaded"]):
            logger.warning(f"Server error detected: {error}")
            return True

        # LiteLLM routing errors
        if "litellm" in error_str or "bad request" in error_str:
            logger.warning(f"LiteLLM error detected: {error}")
            return True

        # Authentication errors (wrong/missing API key)
        if "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
            logger.warning(f"Authentication error detected: {error}")
            return True

        # Connection errors
        if "connection" in error_str or "timeout" in error_str:
            logger.warning(f"Connection error detected: {error}")
            return True

        # Generic provider failures
        if "unable to" in error_str or "api error" in error_str:
             logger.warning(f"Generic API failure detected: {error}")
             return True

        return False

    def get_fallback_message(self, original_task: str, error: Exception) -> str:
        """Generate fallback message when using degraded model.

        Args:
            original_task: The original task that failed
            error: The error that occurred

        Returns:
            User-friendly fallback message
        """
        error_str = str(error)
        error_type = "API error"
        if "429" in error_str or "rate_limit" in error_str.lower():
            error_type = "Rate limit"
        elif "timeout" in error_str.lower():
            error_type = "Timeout"
        elif "connection" in error_str.lower():
            error_type = "Connection issue"
        elif "resource exhausted" in error_str.lower():
            error_type = "Quota exhausted"

        return f"⚠️ *{error_type}* — using backup model. Responses may be simpler than usual."
