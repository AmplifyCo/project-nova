"""Plugin Loader — discovers and loads tool plugins from the plugins/ directory.

Hub-and-spoke architecture:
  Hub  = existing ToolRegistry, PolicyGate, ConversationManager (never touched)
  Spoke = each plugin folder under plugins/ (self-contained, manifest-driven)

Each plugin has:
  - tool.py (or custom module_file): a class extending BaseTool
  - manifest.json: name, risk_map, env_vars, safe_readonly, persona, etc.

PluginLoader is called once at startup via registry.__init__().
Adding a new tool = create folder + manifest.json. Zero hub edits.
"""

import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PLUGINS_DIR = Path(__file__).parent


@dataclass
class PluginManifest:
    """Parsed and validated plugin manifest."""

    name: str
    class_name: str
    plugin_dir: Path
    version: str = "1.0"
    description: str = ""
    module_file: str = "tool.py"
    risk_map: Dict[str, str] = field(default_factory=lambda: {"_default": "write"})
    env_vars: List[str] = field(default_factory=list)
    constructor_args: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    safe_readonly: bool = False
    persona: Optional[str] = None

    @classmethod
    def from_file(cls, manifest_path: Path) -> "PluginManifest":
        """Load and validate a manifest file."""
        with open(manifest_path, "r") as f:
            data = json.load(f)

        for req in ("name", "class_name"):
            if req not in data:
                raise ValueError(f"Manifest missing required field: {req}")

        name = data["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Plugin name must be alphanumeric/underscores: {name}")

        return cls(
            name=name,
            class_name=data["class_name"],
            plugin_dir=manifest_path.parent,
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            module_file=data.get("module_file", "tool.py"),
            risk_map=data.get("risk_map", {"_default": "write"}),
            env_vars=data.get("env_vars", []),
            constructor_args=data.get("constructor_args", {}),
            dependencies=data.get("dependencies", []),
            safe_readonly=data.get("safe_readonly", False),
            persona=data.get("persona"),
        )


class PluginLoader:
    """Discovers and loads tool plugins from the plugins directory."""

    def __init__(self, plugins_dir: Path = None):
        self.plugins_dir = plugins_dir or PLUGINS_DIR
        self._loaded: Dict[str, PluginManifest] = {}

    def discover(self) -> List[PluginManifest]:
        """Scan plugins directory for valid plugin subdirectories.

        A valid plugin has a subdirectory with manifest.json + the module file.
        """
        manifests = []
        if not self.plugins_dir.exists():
            return manifests

        for entry in sorted(self.plugins_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue

            manifest_path = entry / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = PluginManifest.from_file(manifest_path)
                tool_file = entry / manifest.module_file
                if not tool_file.exists():
                    logger.warning(f"Plugin {manifest.name}: {manifest.module_file} not found")
                    continue
                manifests.append(manifest)
            except Exception as e:
                logger.warning(f"Plugin {entry.name}: invalid manifest: {e}")

        return manifests

    def load_all(self, registry) -> int:
        """Load all discovered plugins into the registry.

        Called once from ToolRegistry.__init__().
        Follows the same pattern as _register_*_tool(): check env vars,
        import, instantiate, register. Skip on failure — never crash startup.

        Returns:
            Number of plugins successfully loaded
        """
        manifests = self.discover()
        loaded = 0
        for manifest in manifests:
            if self._load_from_manifest(manifest, registry):
                loaded += 1
        if loaded:
            logger.info(f"Loaded {loaded} plugin(s) from {self.plugins_dir}")
        return loaded

    def reload_all(self, registry) -> Tuple[int, int, List[str]]:
        """Hot-reload: rediscover plugins, load new ones, reload changed ones.

        Returns:
            (new_count, reloaded_count, errors_list)
        """
        manifests = self.discover()
        new_count = 0
        reloaded_count = 0
        errors = []

        discovered_names = {m.name for m in manifests}

        for manifest in manifests:
            if manifest.name in self._loaded:
                # Already loaded — reload it (file may have changed)
                try:
                    self._unload_plugin(manifest.name, registry)
                    if self._load_from_manifest(manifest, registry):
                        reloaded_count += 1
                    else:
                        errors.append(f"{manifest.name}: reload failed")
                except Exception as e:
                    errors.append(f"{manifest.name}: {e}")
            else:
                # New plugin
                try:
                    if self._load_from_manifest(manifest, registry):
                        new_count += 1
                    else:
                        errors.append(f"{manifest.name}: load failed")
                except Exception as e:
                    errors.append(f"{manifest.name}: {e}")

        # Unload plugins whose folders were removed
        removed = set(self._loaded.keys()) - discovered_names
        for name in removed:
            self._unload_plugin(name, registry)
            logger.info(f"Plugin removed (folder deleted): {name}")

        summary = f"Plugins reloaded: {new_count} new, {reloaded_count} updated"
        if removed:
            summary += f", {len(removed)} removed"
        if errors:
            summary += f", {len(errors)} errors"
        logger.info(summary)

        return new_count, reloaded_count, errors

    def reload_plugin(self, name: str, registry) -> Tuple[bool, str]:
        """Hot-reload a single plugin by name.

        Returns:
            (success, message)
        """
        plugin_dir = self.plugins_dir / name
        manifest_path = plugin_dir / "manifest.json"
        if not manifest_path.exists():
            return False, f"Plugin '{name}' not found (no manifest.json)"
        try:
            manifest = PluginManifest.from_file(manifest_path)
        except Exception as e:
            return False, f"Plugin '{name}' manifest error: {e}"

        # Unload if previously loaded
        if name in self._loaded:
            self._unload_plugin(name, registry)

        if self._load_from_manifest(manifest, registry):
            return True, f"Plugin '{name}' reloaded successfully"
        return False, f"Plugin '{name}' failed to load"

    def _unload_plugin(self, name: str, registry):
        """Fully unload a plugin: registry, sys.modules, PolicyGate, internal tracking."""
        # Remove from ToolRegistry
        registry.unregister(name)

        # Remove from sys.modules so importlib loads fresh code
        module_key = f"nova_plugins.{name}"
        if module_key in sys.modules:
            del sys.modules[module_key]

        # Remove from PolicyGate TOOL_RISK_MAP
        try:
            from src.core.nervous_system.policy_gate import TOOL_RISK_MAP
            TOOL_RISK_MAP.pop(name, None)
        except ImportError:
            pass

        # Remove from internal tracking
        self._loaded.pop(name, None)
        logger.info(f"Plugin unloaded: {name}")

    def get_plugin_metadata(self) -> Dict[str, Dict]:
        """Return metadata for all loaded plugins.

        Used by hub components to dynamically extend _SAFE_READONLY_TOOLS,
        _VALID_TOOL_NAMES, and persona detection — without hardcoding.
        """
        return {
            name: {
                "safe_readonly": m.safe_readonly,
                "persona": m.persona,
                "description": m.description,
                "risk_map": m.risk_map,
            }
            for name, m in self._loaded.items()
        }

    def _load_from_manifest(self, manifest: PluginManifest, registry) -> bool:
        """Load a plugin from a validated manifest.

        Steps:
          1. Skip if already loaded
          2. Check env vars (skip if missing — same as _register_*_tool pattern)
          3. Import module dynamically via importlib
          4. Instantiate the tool class
          5. Register in ToolRegistry
          6. Update PolicyGate TOOL_RISK_MAP
        """
        # Credential check (same pattern as existing _register_*_tool methods)
        if manifest.env_vars:
            missing = [v for v in manifest.env_vars if not os.getenv(v)]
            if missing:
                logger.debug(
                    f"Plugin {manifest.name} skipped (missing env: {', '.join(missing)})"
                )
                return False

        try:
            # Dynamic import via importlib
            module_path = manifest.plugin_dir / manifest.module_file
            spec = importlib.util.spec_from_file_location(
                f"nova_plugins.{manifest.name}", str(module_path)
            )
            if spec is None or spec.loader is None:
                logger.error(f"Plugin {manifest.name}: could not create module spec")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Get the tool class
            tool_class = getattr(module, manifest.class_name, None)
            if tool_class is None:
                logger.error(f"Plugin {manifest.name}: class {manifest.class_name} not found")
                return False

            # Build constructor kwargs from env vars
            ctor_kwargs = {}
            for param, env_var in manifest.constructor_args.items():
                ctor_kwargs[param] = os.getenv(env_var, "")

            # Instantiate and validate
            tool = tool_class(**ctor_kwargs)
            if not hasattr(tool, "name") or not hasattr(tool, "execute"):
                logger.error(f"Plugin {manifest.name}: does not implement BaseTool")
                return False

            # Register in ToolRegistry
            registry.register(tool)

            # Update PolicyGate TOOL_RISK_MAP
            self._apply_risk_map(manifest)

            self._loaded[manifest.name] = manifest
            logger.info(f"Plugin loaded: {manifest.name} ({manifest.description})")
            return True

        except Exception as e:
            logger.error(f"Plugin {manifest.name}: load failed: {e}", exc_info=True)
            return False

    @staticmethod
    def _apply_risk_map(manifest: PluginManifest):
        """Inject plugin's risk classification into PolicyGate's TOOL_RISK_MAP."""
        from src.core.nervous_system.policy_gate import TOOL_RISK_MAP, RiskLevel

        level_map = {
            "read": RiskLevel.READ,
            "write": RiskLevel.WRITE,
            "irreversible": RiskLevel.IRREVERSIBLE,
        }
        entry = {}
        for op, level_str in manifest.risk_map.items():
            entry[op] = level_map.get(level_str.lower(), RiskLevel.WRITE)
        TOOL_RISK_MAP[manifest.name] = entry
