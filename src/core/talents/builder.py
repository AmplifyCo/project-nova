#!/usr/bin/env python3
"""
Talent Builder — one-shot agent that builds a new talent
by reading existing patterns and using Claude to generate code.

Usage:
    python -m src.core.talents.builder Post-on-X
    python -m src.core.talents.builder LinkedIn
"""
import os
import sys
import asyncio
import getpass
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.talents.catalog import TalentCatalog
from src.core.agent import AutonomousAgent
from src.core.types import AgentConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_file(path: str) -> str:
    """Read a file and return contents, or empty string if missing."""
    full = PROJECT_ROOT / path
    if full.exists():
        return full.read_text()
    return ""


def build_system_prompt(talent_name: str, talent_config: dict, category: str, key: str) -> str:
    """Build a compact system prompt with patterns for the agent.

    Uses interface summaries instead of full file contents to stay
    within API rate limits (10k tokens/min on lower tiers).
    """
    build_instructions = talent_config.get("build_instructions", "")
    library = talent_config.get("library", "")
    env_vars = talent_config.get("env_vars", [])
    description = talent_config.get("description", "")

    tool_filename = f"{key}_tool.py"

    prompt = f"""You are a talent builder for the Digital Twin system.
Build the "{talent_name}" talent ({description}).

## BUILD INSTRUCTIONS
{build_instructions}

## WHAT TO CREATE

### 1. Tool File: {PROJECT_ROOT}/src/core/tools/{tool_filename}
```python
import os
import logging
from .base import BaseTool
from ..types import ToolResult

class {key.title()}Tool(BaseTool):
    name = "{key}"
    description = "..."
    parameters = {{
        "action": {{"type": "string", "description": "Action to perform"}},
        # ... other params
    }}

    def __init__(self, **credentials):
        self.cred = credentials

    async def execute(self, action: str = None, **kwargs) -> ToolResult:
        try:
            # ... implementation
            return ToolResult(success=True, output="...")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```
- ToolResult has: success (bool), output (str), error (str), metadata (dict)
- Read credentials from env vars: {', '.join(env_vars)}
- Use async/await. Handle errors. Return ToolResult always.

### 2. Update Registry: {PROJECT_ROOT}/src/core/tools/registry.py
Read the file first, then add a method following this exact pattern:
```python
def _register_{key}_tool(self):
    try:
        env_vals = [os.getenv(v) for v in {env_vars}]
        if all(env_vals):
            from .{key}_tool import {key.title()}Tool
            tool = {key.title()}Tool(...)  # pass credentials
            self.register(tool)
            logger.info("Tool registered")
    except Exception as e:
        logger.warning(f"Failed to register: {{e}}")
```
Call `self._register_{key}_tool()` in `__init__()` after existing registrations.

### 3. Update Config: {PROJECT_ROOT}/config/talents.yaml
Read the file, then change the {key} talent entry:
- Set `coming_soon: false`
- Set `enabled: true`

{f"### 4. Add to {PROJECT_ROOT}/requirements.txt: '{library}'" if library else ""}

## RULES
- Read existing files before modifying them
- Use file_operations tool to write files (not bash echo/cat)
- Use absolute paths: {PROJECT_ROOT}/...
- Keep implementation focused and minimal
"""
    return prompt


def prompt_credentials(talent_config: dict) -> dict:
    """Interactively prompt for credentials and return key-value pairs."""
    credential_prompts = talent_config.get("credential_prompts", {})
    env_vars = talent_config.get("env_vars", [])

    if not credential_prompts and not env_vars:
        return {}

    print()
    print("📱 Configure credentials:")
    print()

    credentials = {}
    for var in env_vars:
        prompt_text = credential_prompts.get(var, var)
        if "secret" in var.lower() or "password" in var.lower() or "token" in var.lower():
            value = getpass.getpass(f"  {prompt_text}: ")
        else:
            value = input(f"  {prompt_text}: ").strip()
        if value:
            credentials[var] = value

    return credentials


def save_credentials_to_env(credentials: dict):
    """Append credentials to .env file."""
    env_path = PROJECT_ROOT / ".env"

    # Read existing content
    existing = ""
    if env_path.exists():
        existing = env_path.read_text()

    # Append new credentials
    with open(env_path, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n# --- Added by talent builder ---\n")
        for key, value in credentials.items():
            # Don't duplicate if already set
            if f"{key}=" not in existing:
                f.write(f"{key}={value}\n")

    print("  ✅ Credentials saved to .env")


async def build_talent(talent_name: str):
    """Build a talent using the autonomous agent."""

    # Load catalog
    catalog = TalentCatalog()
    match = catalog.get_talent_by_name(talent_name)

    if match is None:
        print(f"❌ Unknown talent: '{talent_name}'")
        print()
        print("Available talents:")
        catalog.print_status()
        return False

    category, key, talent_config = match
    display_name = talent_config.get("display_name", key)

    print()
    print(f"🔧 Building talent: {display_name}")
    print("━" * 52)
    print()

    # Check if already built (tool file exists = skip build)
    tool_file = PROJECT_ROOT / "src" / "core" / "tools" / f"{key}_tool.py"
    if tool_file.exists():
        print(f"✅ {display_name} talent already exists!")
        print(f"   Tool: {tool_file}")
        print()

        # Still offer to configure credentials
        env_vars = talent_config.get("env_vars", [])
        if env_vars and not all(os.getenv(v) for v in env_vars):
            print(f"🔧 Credentials not configured yet.")
            credentials = prompt_credentials(talent_config)
            if credentials:
                save_credentials_to_env(credentials)
        else:
            print("   Credentials: ✅ configured")
        return True

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment or .env")
        print("   Run: dt-setup core")
        return False

    # Build system prompt
    system_prompt = build_system_prompt(display_name, talent_config, category, key)

    # Create agent config
    config = AgentConfig(
        api_key=api_key,
        default_model="claude-sonnet-4-6",
        max_iterations=15,
    )

    # Create and run agent
    agent = AutonomousAgent(config=config)

    task = f"""Build the "{display_name}" talent for the Digital Twin system.

Follow the system prompt instructions exactly. Create the tool file, update the registry, and update talents.yaml.

Working directory: {PROJECT_ROOT}

After creating files, verify them by reading them back."""

    print(f"🤖 Agent building {display_name}...")
    print(f"   Model: {config.default_model}")
    print(f"   Max iterations: {config.max_iterations}")
    print()

    try:
        result = await agent.run(
            task=task,
            system_prompt=system_prompt,
            max_iterations=config.max_iterations
        )

        print()
        print("━" * 52)

        # Check if tool file was created
        if tool_file.exists():
            print(f"✅ Tool file created: src/core/tools/{key}_tool.py")
        else:
            print(f"⚠️  Tool file not found: src/core/tools/{key}_tool.py")

        # Check if registry was updated
        registry_content = load_file("src/core/tools/registry.py")
        if f"_register_{key}_tool" in registry_content:
            print(f"✅ Registry updated")
        else:
            print(f"⚠️  Registry may need manual update")

        # Check if talents.yaml was updated
        catalog_fresh = TalentCatalog()
        match_fresh = catalog_fresh.get_talent_by_name(talent_name)
        if match_fresh and not match_fresh[2].get("coming_soon"):
            print(f"✅ Config updated")
        else:
            print(f"⚠️  Config may need manual update")

        print()

        # Prompt for credentials
        credentials = prompt_credentials(talent_config)
        if credentials:
            save_credentials_to_env(credentials)

        # Install library if needed
        library = talent_config.get("library")
        if library:
            print(f"\n📦 Installing {library}...")
            os.system(f"pip install {library}")
            # Add to requirements.txt if not there
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.exists():
                req_content = req_path.read_text()
                if library not in req_content:
                    with open(req_path, "a") as f:
                        f.write(f"\n{library}\n")
                    print(f"   Added {library} to requirements.txt")

        print()
        print(f"✅ {display_name} talent built!")
        print(f"   Restart the bot to activate: dt-setup restart")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        logger.exception("Build failed")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.core.talents.builder <talent-name>")
        print()
        print("Examples:")
        print("  python -m src.core.talents.builder Post-on-X")
        print("  python -m src.core.talents.builder LinkedIn")
        print()
        # Show available talents
        catalog = TalentCatalog()
        catalog.print_status()
        return

    talent_name = " ".join(sys.argv[1:])
    success = asyncio.run(build_talent(talent_name))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
