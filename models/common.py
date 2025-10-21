"""Common models shared across all services"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class ProviderKind(Enum):
    """Unified provider types for both agent and audio services"""
    # Agent providers
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    # Audio providers
    ELEVENLABS = "elevenlabs"


class Settings(BaseModel):
    """Base settings for all service providers"""
    provider: ProviderKind = Field(..., description="The provider to use")
    api_key: str = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Optional custom base URL for the provider")

