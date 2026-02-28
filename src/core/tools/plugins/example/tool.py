"""Example plugin to verify the plugin system works."""

from src.core.tools.base import BaseTool
from src.core.types import ToolResult


class ExampleTool(BaseTool):
    """Example plugin tool — returns a test response."""

    name = "example_plugin"
    description = "Example plugin to verify the plugin system works. Call with operation='ping'."
    parameters = {
        "operation": {
            "type": "string",
            "description": "Operation to perform: 'ping' returns a test response",
            "enum": ["ping"],
        },
    }

    async def execute(self, operation: str = "ping", **kwargs) -> ToolResult:
        if operation == "ping":
            return ToolResult(success=True, output="Plugin system is working!")
        return ToolResult(success=False, error=f"Unknown operation: {operation}")
