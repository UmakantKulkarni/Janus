"""Repository path utilities.

This module centralizes loading of the project configuration and provides
helpers for resolving paths relative to the project root.  The root directory
is configured in ``config/default.yaml`` under ``paths.project_root``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

@lru_cache()
def load_repo_config() -> dict:
    """Load and cache repository configuration from YAML."""
    config_env = os.getenv("JANUS_CONFIG")
    config_path = Path(config_env) if config_env else DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

@lru_cache()
def project_root() -> Path:
    """Return the root directory of the project.

    If ``paths.project_root`` is omitted from the configuration the directory
    containing the configuration file is used instead.
    """
    cfg = load_repo_config()
    root = cfg.get("paths", {}).get("project_root")
    return Path(root).expanduser().resolve() if root else DEFAULT_CONFIG.parent.parent

def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against :func:`project_root`.

    Absolute paths are returned unchanged; relative paths are joined with the
    project root.
    """
    p = Path(path)
    return p if p.is_absolute() else project_root() / p
