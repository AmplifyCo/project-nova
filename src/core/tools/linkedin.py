"""LinkedIn Tool — post text and articles to LinkedIn on principal's behalf.

Uses LinkedIn UGC Posts API (v2) with OAuth 2.0 3-legged token.
This is the official API — no browser automation, no bot risk.

Setup (one-time):
    python scripts/linkedin_auth.py

Environment variables required:
    LINKEDIN_ACCESS_TOKEN   — OAuth access token (from setup script)
    LINKEDIN_PERSON_URN     — urn:li:person:XXXXXXX (from setup script)

Rate limits:
    150 posts/day per member (official LinkedIn limit)
"""

import logging
from typing import Optional

import aiohttp

from .base import BaseTool
from ..types import ToolResult

logger = logging.getLogger(__name__)

_UGC_POSTS_URL = "https://api.linkedin.com/v2/ugcPosts"
_RESTLI_HEADER = {"X-Restli-Protocol-Version": "2.0.0"}


class LinkedInTool(BaseTool):
    """Tool to post to LinkedIn on principal's behalf via the official API.

    Supports text posts and article/URL shares. Uses OAuth 2.0 — no browser
    automation, no ToS violation, no ban risk.
    """

    name = "linkedin"
    description = (
        "Tool to post to LinkedIn on principal's behalf using the official LinkedIn API. "
        "Use for: publishing professional insights, sharing articles with commentary, "
        "posting thoughts on AI/tech trends to the professional network. "
        "Operations: 'post_text' (text only), 'post_article' (URL + commentary). "
        "Do NOT use for casual social posts — LinkedIn is professional context only."
    )
    parameters = {
        "operation": {
            "type": "string",
            "enum": ["post_text", "post_article"],
            "description": (
                "'post_text': publish a text-only LinkedIn post. "
                "'post_article': share a URL with commentary (and optional title)."
            ),
        },
        "text": {
            "type": "string",
            "description": (
                "Post commentary / text. Required for all operations. "
                "LinkedIn best practice: 1-3 short paragraphs, end with a question or insight. "
                "Hashtags optional but effective (3-5 max)."
            ),
        },
        "url": {
            "type": "string",
            "description": "URL to share. Required for 'post_article'.",
        },
        "title": {
            "type": "string",
            "description": "Optional title override for the shared article.",
        },
        "visibility": {
            "type": "string",
            "enum": ["PUBLIC", "CONNECTIONS"],
            "description": "Who can see the post. Default: PUBLIC.",
        },
    }

    def __init__(self, access_token: str, person_urn: str):
        """
        Args:
            access_token: LinkedIn OAuth 2.0 access token
            person_urn: LinkedIn person URN, e.g. 'urn:li:person:XXXXXXX'
        """
        self.access_token = access_token
        self.person_urn = person_urn

    async def execute(
        self,
        operation: str,
        text: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        visibility: str = "PUBLIC",
        **kwargs,
    ) -> ToolResult:
        if not text or not text.strip():
            return ToolResult(success=False, error="'text' is required")

        if operation == "post_text":
            return await self._post_text(text.strip(), visibility)
        elif operation == "post_article":
            if not url or not url.strip():
                return ToolResult(success=False, error="'url' is required for post_article")
            return await self._post_article(text.strip(), url.strip(), title, visibility)
        else:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _post_text(self, text: str, visibility: str) -> ToolResult:
        body = {
            "author": self.person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility
            },
        }
        return await self._call_api(body)

    async def _post_article(
        self,
        text: str,
        url: str,
        title: Optional[str],
        visibility: str,
    ) -> ToolResult:
        media: dict = {"status": "READY", "originalUrl": url}
        if title:
            media["title"] = {"text": title}

        body = {
            "author": self.person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "ARTICLE",
                    "media": [media],
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility
            },
        }
        return await self._call_api(body)

    async def _call_api(self, body: dict) -> ToolResult:
        headers = {
            **_RESTLI_HEADER,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _UGC_POSTS_URL, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 201:
                        post_id = resp.headers.get("X-RestLi-Id", "unknown")
                        logger.info(f"LinkedIn post created: {post_id}")
                        return ToolResult(
                            success=True,
                            output=f"LinkedIn post published (ID: {post_id})",
                            metadata={"post_id": post_id},
                        )
                    else:
                        error_body = await resp.text()
                        logger.error(f"LinkedIn API {resp.status}: {error_body[:300]}")
                        # Token expired = 401; provide clear action
                        if resp.status == 401:
                            return ToolResult(
                                success=False,
                                error=(
                                    "LinkedIn access token expired. "
                                    "Run: python scripts/linkedin_auth.py to refresh."
                                ),
                            )
                        return ToolResult(
                            success=False,
                            error=f"LinkedIn API error {resp.status}: {error_body[:200]}",
                        )
        except aiohttp.ClientError as e:
            logger.error(f"LinkedIn network error: {e}")
            return ToolResult(success=False, error=f"LinkedIn network error: {e}")
        except Exception as e:
            logger.error(f"LinkedIn post failed: {e}", exc_info=True)
            return ToolResult(success=False, error=f"LinkedIn post failed: {e}")
