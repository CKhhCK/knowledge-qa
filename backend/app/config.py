from __future__ import annotations
"""
Application configuration via Pydantic Settings.

All configuration is driven by environment variables with sensible defaults.
Uses pydantic-settings for automatic .env file loading and validation.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Configuration ---
    llm_model_id: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.7
    llm_timeout: int = 60

    # --- Agent Configuration ---
    max_react_steps: int = 5
    max_reflection_iterations: int = 1
    reflection_enabled: bool = True
    react_timeout_seconds: int = 120

    # --- Search API Keys ---
    tavily_api_key: str = ""
    serpapi_api_key: str = ""

    # --- Knowledge Base Configuration ---
    knowledge_base_path: str = "./data/knowledge_base"
    rag_namespace: str = "default"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_enable_advanced_search: bool = True

    # --- Memory Configuration ---
    memory_user_id: str = "qa_system"
    memory_types: str = "working,episodic,semantic"
    memory_working_capacity: int = 100
    memory_working_ttl_minutes: int = 120

    @property
    def memory_type_list(self) -> list[str]:
        return [t.strip() for t in self.memory_types.split(",") if t.strip()]

    # --- Qdrant Configuration (Vector Database) ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # --- Server Configuration ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # --- Session Configuration ---
    session_max_count: int = 1000
    session_ttl_seconds: int = 3600  # 1 hour

    # --- Rate Limiting ---
    rate_limit_per_session: int = 300  # requests per minute

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton Settings instance (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
