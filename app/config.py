"""
Application configuration loaded from environment variables.

All settings are centralised here so no module reads os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables / .env."""

    # ── Paths ────────────────────────────────────────────────────────────
    artifact_dir: str = Field(
        default=str(Path(__file__).resolve().parent.parent / "artifacts"),
        description="Absolute path to the directory containing model artifacts.",
    )

    # ── Hugging Face ─────────────────────────────────────────────────────
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face API token for embedding generation via Inference API.",
    )

    # ── API Security ─────────────────────────────────────────────────────
    api_key: str = Field(
        default="",
        description="Expected API key for authenticated endpoints. "
                    "Leave empty to disable key checking.",
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Max requests per rate-limit window.",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Duration of the rate-limit sliding window in seconds.",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins.",
    )

    # ── Server ───────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    log_level: str = Field(default="info")

    # ── Versioning ───────────────────────────────────────────────────────
    model_version: str = Field(default="1.0.0")
    build_timestamp: str = Field(default="2026-09-01T00:00:00Z")

    # ── Batch ────────────────────────────────────────────────────────────
    max_batch_size: int = Field(
        default=500,
        description="Maximum number of records in a single batch request.",
    )

    model_config = {
        "env_prefix": "RM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Helpers ──────────────────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
