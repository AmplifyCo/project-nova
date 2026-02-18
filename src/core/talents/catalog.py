"""
Talent Catalog — reads config/talents.yaml and checks which
talents are active based on environment variables.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Category display config
CATEGORY_META = {
    "communication":  {"icon": "📡", "label": "Communication"},
    "social_media":   {"icon": "📱", "label": "Social Media"},
    "productivity":   {"icon": "📅", "label": "Productivity"},
    "research":       {"icon": "🔍", "label": "Research & Information"},
    "files_and_code": {"icon": "💻", "label": "Files & Code"},
    "finance":        {"icon": "💰", "label": "Finance"},
}


class TalentCatalog:
    """
    Reads talents.yaml and determines status of each talent
    by checking required environment variables.
    """

    STATUS_ACTIVE = "active"
    STATUS_NOT_CONFIGURED = "not_configured"
    STATUS_COMING_SOON = "coming_soon"

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Find config/talents.yaml relative to project root
            here = Path(__file__).resolve()
            self.project_root = here.parents[3]  # src/core/talents/ -> project root
            config_path = self.project_root / "config" / "talents.yaml"
        else:
            self.project_root = Path(config_path).resolve().parents[1]

        self.config_path = Path(config_path)
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get_status(self, talent_config: Dict[str, Any]) -> str:
        if talent_config.get("coming_soon"):
            return self.STATUS_COMING_SOON

        if talent_config.get("always_available"):
            return self.STATUS_ACTIVE

        env_vars = talent_config.get("env_vars", [])
        if env_vars and all(os.getenv(v) for v in env_vars):
            return self.STATUS_ACTIVE

        if env_vars:
            return self.STATUS_NOT_CONFIGURED

        return self.STATUS_ACTIVE

    def get_talent_by_name(self, name: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Fuzzy-match a talent by display name, key, or hyphenated name.
        Returns (category, key, config) or None.

        Examples:
            "Post-on-X" → ("social_media", "x", {...})
            "email" → ("communication", "email", {...})
            "LinkedIn" → ("social_media", "linkedin", {...})
        """
        normalized = name.lower().replace("-", " ").replace("_", " ").strip()

        for category, talents in self._config.items():
            for key, cfg in talents.items():
                display = cfg.get("display_name", key).lower()
                # Match against: key, display name, or normalized input
                if (key == normalized
                        or display == normalized
                        or normalized in display
                        or display.replace(" ", "") == normalized.replace(" ", "")):
                    return (category, key, cfg)
        return None

    def get_all(self) -> Dict[str, List[Dict]]:
        """Return all talents grouped by category with status."""
        result = {}
        for category, talents in self._config.items():
            meta = CATEGORY_META.get(category, {"icon": "•", "label": category.replace("_", " ").title()})
            entries = []
            for key, cfg in talents.items():
                status = self.get_status(cfg)
                entries.append({
                    "key": key,
                    "display_name": cfg.get("display_name", key),
                    "description": cfg.get("description", ""),
                    "status": status,
                    "dt_setup_cmd": cfg.get("dt_setup_cmd", ""),
                })
            result[category] = {"meta": meta, "talents": entries}
        return result

    def print_status(self):
        """Print a formatted talent status table."""
        all_talents = self.get_all()
        total_active = 0
        total_coming_soon = 0
        total_not_configured = 0

        print()
        print("🧠 Digital Twin — Talents")
        print("━" * 52)

        for category, data in all_talents.items():
            meta = data["meta"]
            print(f"\n{meta['icon']}  {meta['label']}")
            for t in data["talents"]:
                if t["status"] == self.STATUS_ACTIVE:
                    icon = "✅"
                    total_active += 1
                elif t["status"] == self.STATUS_NOT_CONFIGURED:
                    icon = "🔧"
                    total_not_configured += 1
                else:
                    icon = "⚙️ "
                    total_coming_soon += 1

                name = t["display_name"].ljust(22)
                desc = t["description"] or "Coming soon"
                print(f"  {icon} {name}{desc}")

                if t["status"] == self.STATUS_NOT_CONFIGURED and t["dt_setup_cmd"]:
                    print(f"       Configure: {t['dt_setup_cmd']}")

        print()
        print("━" * 52)
        parts = []
        if total_active:
            parts.append(f"✅ Active: {total_active}")
        if total_not_configured:
            parts.append(f"🔧 Not configured: {total_not_configured}")
        if total_coming_soon:
            parts.append(f"⚙️  Coming soon: {total_coming_soon}")
        print("  " + "  |  ".join(parts))
        print()
