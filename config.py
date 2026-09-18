"""Application configuration loaded from the environment and local .env file."""

from __future__ import annotations

import os
from pathlib import Path


ENV_FILE = Path(__file__).with_name(".env")


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs without replacing existing environment values."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and not os.getenv(key):
            os.environ[key] = value


def get_api_key(provider: str) -> str:
    """Return the configured API key for an answer-generator provider."""
    environment_variable = {
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY",
    }.get(provider)
    return os.getenv(environment_variable, "").strip() if environment_variable else ""
