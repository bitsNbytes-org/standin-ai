"""Simple configuration management."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    
    # Model Configuration  
    text_model: str = Field(default="gemini-2.5-flash-lite")
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 