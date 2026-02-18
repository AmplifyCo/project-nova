"""Tool registry for managing available tools."""

import logging
from typing import List, Dict, Any, Optional
from .base import BaseTool
from .bash import BashTool
from .file import FileTool
from .web import WebTool
from .browser import BrowserTool
from ..types import ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of all available tools for the agent."""

    # Tools that require semantic validation (potentially dangerous)
    TOOLS_REQUIRING_VALIDATION = ['bash', 'file_write']

    def __init__(self, config: Dict[str, Any] = None, security_guard=None):
        """Initialize tool registry with default tools.

        Args:
            config: Configuration dictionary for tools
            security_guard: Optional LLMSecurityGuard for semantic validation
        """
        self.tools: Dict[str, BaseTool] = {}
        self.config = config or {}
        self.security_guard = security_guard
        self.semantic_validation_enabled = self.config.get('semantic_validation', {}).get('enabled', False)

        # Get safety config
        safety_config = self.config.get('safety', {})

        # Register default tools with configuration
        self.register(BashTool(
            allowed_commands=safety_config.get('allowed_commands', []),
            blocked_commands=safety_config.get('blocked_commands', []),
            allow_sudo=safety_config.get('allow_sudo', False),
            allowed_sudo_commands=safety_config.get('allowed_sudo_commands', [])
        ))
        self.register(FileTool())
        self.register(WebTool())
        self.register(BrowserTool())

    def register(self, tool: BaseTool):
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for Claude API.

        Returns:
            List of tool definitions in Anthropic format
        """
        return [tool.to_anthropic_tool() for tool in self.tools.values()]

    async def execute_tool(
        self,
        tool_name: str,
        user_message: str = "",
        llm_client=None,
        **params
    ) -> ToolResult:
        """Execute a tool by name with optional semantic validation.

        Args:
            tool_name: Name of tool to execute
            user_message: Original user message (for semantic validation)
            llm_client: LLM client for semantic validation
            **params: Tool parameters

        Returns:
            ToolResult from tool execution
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        # ========================================================================
        # LAYER 11: SEMANTIC VALIDATION (for dangerous tools only)
        # ========================================================================
        if (self.semantic_validation_enabled and
            self.security_guard and
            tool_name in self.TOOLS_REQUIRING_VALIDATION and
            user_message and
            llm_client):

            try:
                is_valid, reason = await self.security_guard.validate_tool_use_semantic(
                    message=user_message,
                    tool_name=tool_name,
                    tool_args=params,
                    llm_client=llm_client
                )

                if not is_valid:
                    logger.warning(
                        f"🚨 SEMANTIC VALIDATION FAILED - Tool: {tool_name}, "
                        f"Reason: {reason}"
                    )
                    return ToolResult(
                        success=False,
                        error=f"Security validation failed: {reason}"
                    )
            except Exception as e:
                logger.debug(f"Semantic validation error (allowing): {e}")
                # Fail open - don't block legitimate use if validation fails

        try:
            return await tool.execute(**params)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                error=f"Tool execution error: {str(e)}"
            )

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())
