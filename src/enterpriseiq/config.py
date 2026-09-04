"""Central application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    app_name: str = "EnterpriseIQ"
    app_environment: str = "development"
    app_log_level: str = "INFO"
    app_version: str = "0.6.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Create and cache one Settings instance."""

    return Settings()
