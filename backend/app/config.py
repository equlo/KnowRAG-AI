"""
config.py - All tunable settings for the backend live in ONE place.

Values are read (in priority order) from:
  1. real environment variables (e.g. `export KNOWRAG_TOP_K=8`),
  2. a `.env` file in the backend folder,
  3. the defaults written below.

Every variable name starts with `KNOWRAG_` so it cannot clash with variables
that other tools set on your machine (e.g. a generic `TOP_K` or `CLAUDE_EFFORT`).
The one exception is ANTHROPIC_API_KEY, which keeps its standard name.
"""

# `Path` gives us an object-oriented, cross-platform way to work with file paths.
from pathlib import Path

# `Literal` restricts a field to a fixed set of allowed string values.
from typing import Literal

# `Field` lets us attach validation rules (like "must be > 0") to a setting.
# `SecretStr` hides a value when printed/logged (it shows as '**********').
# `model_validator` runs a check after all fields have been loaded.
from pydantic import Field, SecretStr, model_validator

# `BaseSettings` is a Pydantic model that fills itself from environment variables.
# `SettingsConfigDict` configures *how* it reads them (which .env file, etc.).
from pydantic_settings import BaseSettings, SettingsConfigDict


# Every attribute of this class becomes one setting. The environment variable
# name is the prefix + attribute name in UPPER CASE (e.g. `top_k` <- `KNOWRAG_TOP_K`).
class Settings(BaseSettings):
    # Read variables starting with KNOWRAG_, also from a `.env` file, and silently
    # ignore any other variables in that file.
    model_config = SettingsConfigDict(env_prefix="KNOWRAG_", env_file=".env", extra="ignore")

    # ----- Generative AI (Claude) settings -----

    # The Claude API key. `validation_alias` makes it read the standard
    # ANTHROPIC_API_KEY name (no prefix). If it is unset, the Anthropic SDK falls
    # back to its own lookup (environment variable or `ant auth login` profile).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # The Claude model ID used to write answers.
    claude_model: str = "claude-opus-5-5"
    # Reasoning effort; only these five strings are accepted by the API.
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Ceiling on generated tokens per answer (must be at least 1).
    claude_max_tokens: int = Field(default=64000, ge=1)
    # Whether to ask the API to retry declined requests on a fallback model.
    claude_refusal_fallback: bool = True
    # Whether follow-up questions are rewritten into standalone search queries.
    query_rewrite: bool = True

    # ----- Retrieval settings -----

    # Which embedding backend turns text into vectors.
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    # Name of the FastEmbed model (ignored by the "hash" provider).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Maximum characters per chunk (at least 100 so chunks carry real meaning).
    chunk_size: int = Field(default=1000, ge=100)
    # Characters shared between consecutive chunks (0 = no overlap).
    chunk_overlap: int = Field(default=200, ge=0)
    # Number of chunks retrieved per question (between 1 and 20).
    top_k: int = Field(default=5, ge=1, le=20)

    # ----- Server / storage settings -----

    # Folder that holds the SQLite database file.
    data_dir: Path = Path("./data")
    # Upload size limit in megabytes.
    max_upload_mb: int = Field(default=20, ge=1)
    # Comma-separated list of allowed browser origins for CORS.
    cors_origins: str = "http://localhost:5173"

    # Cross-field check: the overlap must be smaller than the chunk itself,
    # otherwise the chunker could never move forward through the text.
    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        # Fail at startup with a clear message instead of on the first upload.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("KNOWRAG_CHUNK_OVERLAP must be smaller than KNOWRAG_CHUNK_SIZE")
        # Validators must return the (unchanged) object.
        return self

    # A computed (read-only) property: the full path of the database file.
    @property
    def db_path(self) -> Path:
        # Join the data folder and the file name, e.g. ./data/knowrag.db
        return self.data_dir / "knowrag.db"

    # A computed property: CORS origins as a Python list instead of one string.
    @property
    def cors_origin_list(self) -> list[str]:
        # Split on commas, strip spaces, and drop empty entries.
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
