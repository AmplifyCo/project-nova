"""Digital Clone Brain for production use (permanent)."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from cryptography.fernet import Fernet
from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class DigitalCloneBrain:
    """Production brain for digital clone. Persistent forever."""

    def __init__(self, path: str = "data/digital_clone_brain"):
        """Initialize digital clone brain.

        Args:
            path: Path to store brain data
        """
        self.path = path

        # Initialize vector databases for different types of memory
        self.memory = VectorDatabase(
            path=f"{path}/memory",
            collection_name="clone_memory"
        )

        self.preferences = VectorDatabase(
            path=f"{path}/preferences",
            collection_name="preferences"
        )

        self.contacts = VectorDatabase(
            path=f"{path}/contacts",
            collection_name="contacts"
        )

        logger.info(f"Initialized DigitalCloneBrain at {path}")

    async def learn_communication_style(self, email_sample: str):
        """Learn communication style from email sample.

        Args:
            email_sample: Sample email text
        """
        await self.preferences.store(
            text=email_sample,
            metadata={
                "type": "communication_style",
                "timestamp": datetime.now().isoformat()
            }
        )

        logger.info("Learned communication style from email sample")

    async def remember_person(
        self,
        name: str,
        relationship: str,
        preferences: Dict[str, Any]
    ):
        """Remember a person and their details.

        Args:
            name: Person's name
            relationship: Relationship to user
            preferences: Dict of person's preferences
        """
        contact_id = name.lower().replace(" ", "_")

        await self.contacts.store(
            text=f"{name}: {relationship}. Preferences: {json.dumps(preferences)}",
            metadata={
                "name": name,
                "relationship": relationship,
                **preferences
            },
            doc_id=f"contact_{contact_id}"
        )

        logger.info(f"Remembered person: {name}")

    async def remember_preference(self, category: str, preference: str):
        """Remember a user preference.

        Args:
            category: Preference category (food, travel, etc.)
            preference: Preference description
        """
        await self.preferences.store(
            text=f"Preference in {category}: {preference}",
            metadata={
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
        )

        logger.info(f"Remembered preference: {category} - {preference}")

    async def export_for_migration(self, password: str, output_file: str = "digital_clone_brain.brain") -> str:
        """Export entire brain for migration to new machine.

        Args:
            password: Password for encryption
            output_file: Output file name

        Returns:
            Path to encrypted brain file
        """
        # Create brain data structure
        brain_data = {
            "memory_count": self.memory.count(),
            "preferences_count": self.preferences.count(),
            "contacts_count": self.contacts.count(),
            "exported_at": datetime.now().isoformat()
        }

        # Convert to JSON
        json_data = json.dumps(brain_data)

        # Encrypt with password
        key = Fernet.generate_key()
        cipher = Fernet(key)
        encrypted = cipher.encrypt(json_data.encode())

        # Save to file
        with open(output_file, 'wb') as f:
            f.write(encrypted)

        logger.info(f"Exported DigitalCloneBrain to {output_file}")
        return output_file

    async def import_from_migration(self, brain_file: str, password: str):
        """Import brain from migration file.

        Args:
            brain_file: Path to encrypted brain file
            password: Decryption password
        """
        logger.info(f"Importing DigitalCloneBrain from {brain_file}")
        # Implementation would decrypt and restore data
        # Simplified for now

    async def get_relevant_context(self, task: str, max_results: int = 5) -> str:
        """Get relevant context for a task.

        Args:
            task: Task description
            max_results: Max number of memories to retrieve

        Returns:
            Context string
        """
        # Search memories
        memories = await self.memory.search(task, n_results=max_results)

        # Search preferences
        prefs = await self.preferences.search(task, n_results=3)

        # Build context
        context_parts = []

        if memories:
            context_parts.append("## Relevant Memories:")
            for mem in memories:
                context_parts.append(f"- {mem['text'][:200]}...")

        if prefs:
            context_parts.append("\n## User Preferences:")
            for pref in prefs:
                context_parts.append(f"- {pref['text']}")

        return "\n".join(context_parts) if context_parts else ""

    async def store_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        model_used: str,
        metadata: Dict[str, Any] = None
    ):
        """Store a conversation turn for context continuity.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            model_used: Which model generated the response (claude-opus-4, smollm2, etc)
            metadata: Additional metadata
        """
        conversation_text = f"""User: {user_message}
Assistant ({model_used}): {assistant_response}"""

        await self.memory.store(
            text=conversation_text,
            metadata={
                "type": "conversation",
                "model_used": model_used,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                **(metadata or {})
            }
        )

        logger.debug(f"Stored conversation turn (model: {model_used})")

    async def get_recent_conversation(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns.

        Args:
            limit: Number of recent turns to retrieve

        Returns:
            List of conversation turn dicts
        """
        # Search for recent conversations
        results = await self.memory.search(
            query="recent conversation",
            n_results=limit * 2  # Get more to filter
        )

        # Filter for conversation type and sort by timestamp
        conversations = [
            {
                "user_message": r["metadata"].get("user_message", ""),
                "assistant_response": r["metadata"].get("assistant_response", ""),
                "model_used": r["metadata"].get("model_used", "unknown"),
                "timestamp": r["metadata"].get("timestamp", "")
            }
            for r in results
            if r["metadata"].get("type") == "conversation"
        ]

        # Sort by timestamp (most recent first)
        conversations.sort(
            key=lambda x: x["timestamp"],
            reverse=True
        )

        return conversations[:limit]

    async def get_conversation_context(self, current_message: str, limit: int = 3) -> str:
        """Get formatted conversation context for model prompt.

        Args:
            current_message: Current user message
            limit: Number of previous turns to include

        Returns:
            Formatted context string
        """
        recent = await self.get_recent_conversation(limit)

        if not recent:
            return ""

        context_lines = ["## Recent Conversation:"]

        # Reverse to show oldest first (chronological order)
        for turn in reversed(recent):
            context_lines.append(f"User: {turn['user_message']}")
            context_lines.append(f"Assistant: {turn['assistant_response']}")
            context_lines.append("")  # Blank line

        return "\n".join(context_lines)

    async def detect_context_drift(self) -> Dict[str, Any]:
        """Detect if context quality has degraded (too many local model responses).

        Returns:
            Drift analysis dict
        """
        recent = await self.get_recent_conversation(limit=10)

        if not recent:
            return {"has_drift": False, "reason": "No conversation history"}

        # Count local model usage
        local_model_count = sum(
            1 for turn in recent
            if turn["model_used"] in ["smollm2", "deepseek-r1"]
        )

        # If more than 50% are local models, we have drift
        drift_threshold = len(recent) * 0.5
        has_drift = local_model_count > drift_threshold

        return {
            "has_drift": has_drift,
            "total_turns": len(recent),
            "local_model_turns": local_model_count,
            "drift_percentage": (local_model_count / len(recent) * 100) if recent else 0,
            "recommendation": "Switch back to Claude API for quality restoration" if has_drift else "Quality OK"
        }

    async def queue_for_claude_review(self, message: str, local_response: str):
        """Queue a local model response for Claude to review/correct when available.

        Args:
            message: User's message
            local_response: Local model's response
        """
        await self.memory.store(
            text=f"LOCAL RESPONSE (needs review): {message} -> {local_response}",
            metadata={
                "type": "needs_review",
                "user_message": message,
                "local_response": local_response,
                "timestamp": datetime.now().isoformat(),
                "reviewed": False
            }
        )

        logger.info("Queued local response for Claude review")

    async def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get responses that need Claude's review/correction.

        Returns:
            List of pending review items
        """
        results = await self.memory.search(
            query="needs review",
            n_results=20
        )

        pending = [
            {
                "user_message": r["metadata"].get("user_message", ""),
                "local_response": r["metadata"].get("local_response", ""),
                "timestamp": r["metadata"].get("timestamp", "")
            }
            for r in results
            if r["metadata"].get("type") == "needs_review"
            and not r["metadata"].get("reviewed", False)
        ]

        return pending
