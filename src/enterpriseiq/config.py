"""Central application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from enterpriseiq import __version__


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    app_name: str = "EnterpriseIQ"
    app_environment: str = "development"
    app_log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    openai_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=120,
    )
    openai_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )
    openai_max_output_tokens: int = Field(
        default=900,
        ge=100,
        le=4000,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @property
    def app_version(self) -> str:
        """Return the application package version."""

        return __version__


@lru_cache
def get_settings() -> Settings:
    """Create and cache one Settings instance."""

    return Settings()
