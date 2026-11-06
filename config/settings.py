"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet déterminée de manière absolue
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application settings loaded from environment and .env file.

    Attributes:
        gemini_api_key: Secret API key for Google AI Studio.
        gemini_model: Gemini model identifier for job evaluation.
        match_score_threshold: Minimum score (0-100) to trigger human review.
        telegram_bot_token: Bot authentication token for Telegram alerts.
        telegram_chat_id: Target chat identifier for Telegram messages.
        database_path: Filesystem path to SQLite database.
        candidate_profile_path: Filesystem path to candidate Markdown profile.
        resume_pdf_path: Filesystem path to candidate resume in PDF format.
        chrome_user_data_dir: Filesystem path to persistent Chrome user profile.
    """

    model_config = SettingsConfigDict(
        env_file=f"{PROJECT_ROOT}/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Configuration Gemini LLM
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google AI Studio API key.",
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        description="Target Gemini model identifier.",
    )

    # Seuil d'éligibilité pour le module Human-in-the-Loop
    match_score_threshold: int = Field(
        default=75,
        ge=0,
        le=100,
        description="Minimum score required to notify the user.",
    )

    # Configuration Telegram Bot
    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        description="Telegram bot token provided by BotFather.",
    )
    telegram_chat_id: int | None = Field(
        default=None,
        description="Authorized Telegram chat ID for notifications.",
    )

    # Identifiants API France Travail (francetravail.io)
    france_travail_client_id: SecretStr = Field(
        default=SecretStr(""),
        description="France Travail API client ID.",
    )
    france_travail_client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="France Travail API client secret.",
    )

    @field_validator("telegram_chat_id", mode="before")
    @classmethod
    def sanitize_optional_chat_id(cls, value: object) -> int | None:
        """Parse empty strings into None for optional integer chat ID.

        Args:
            value: Raw value from environment or configuration.

        Returns:
            int | None: Parsed integer or None if empty.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, (int, str)):
            return int(value)
        raise ValueError(f"Invalid telegram_chat_id: {value}")

    # Chemins des fichiers et répertoires de données
    profile_dir: Path = Field(
        default=PROJECT_ROOT / "Profile",
        description="Unified candidate profile folder (CV, criteria, profile.md).",
    )
    database_path: Path = Field(
        default=PROJECT_ROOT / "data" / "jobs.db",
        description="SQLite database path.",
    )
    candidate_profile_path: Path = Field(
        default=PROJECT_ROOT / "config" / "profile.md",
        description="Candidate profile markdown path.",
    )
    search_criteria_path: Path = Field(
        default=PROJECT_ROOT / "Profile" / "search_criteria.txt",
        description="Search criteria text configuration path.",
    )
    resume_pdf_path: Path = Field(
        default=PROJECT_ROOT / "Profile" / "Resume Jérémy Foeller.pdf",
        description="Candidate resume PDF file path.",
    )
    chrome_user_data_dir: Path = Field(
        default=Path.home() / ".config" / "job_hunter" / "chrome_profile",
        description="Persistent Chrome user data directory.",
    )

    @field_validator(
        "profile_dir",
        "database_path",
        "candidate_profile_path",
        "search_criteria_path",
        "resume_pdf_path",
        "chrome_user_data_dir",
        mode="after",
    )
    @classmethod
    def resolve_to_absolute_path(cls, path_value: Path) -> Path:
        """Resolve relative paths against the project root.

        Args:
            path_value: Path specified either relatively or absolutely.

        Returns:
            Path: Resolved absolute path.
        """
        # Si le chemin commence par ~, on résout le dossier utilisateur
        expanded = path_value.expanduser()
        if not expanded.is_absolute():
            # Résolution relative à la racine du projet
            return (PROJECT_ROOT / expanded).resolve()
        return expanded.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instantiate and cache the application settings.

    Returns:
        Settings: Singleton instance of the application settings.
    """
    # Instanciation unique mise en cache
    return Settings()
