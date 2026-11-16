"""Load and validate application settings."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    log_level: str = DEFAULT_LOG_LEVEL


def load_config(path: Path | None = None) -> ApplicationConfig:
    """Read an optional UTF-8 TOML file and validate its settings."""
    if path is None:
        return ApplicationConfig()

    with path.open("rb") as config_file:
        settings = tomllib.load(config_file)

    unknown_keys = settings.keys() - {"log_level"}
    if unknown_keys:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown_keys))}")

    log_level = settings.get("log_level", DEFAULT_LOG_LEVEL)
    if not isinstance(log_level, str) or log_level.upper() not in LOG_LEVELS:
        raise ValueError(f"log_level must be one of: {', '.join(LOG_LEVELS)}")
    return ApplicationConfig(log_level=log_level.upper())
