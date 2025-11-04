"""Common models shared across all services"""

from pydantic import BaseModel, Field, SecretStr
from enum import Enum
from typing import Optional


class BaseRequest(BaseModel):
    """Base request model that all service requests inherit from"""
    provider_uid: str = Field(..., description="The unique identifier of the provider configuration to use")


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
    uid: str = Field(..., description="The unique identifier of the provider configuration")
    provider: ProviderKind = Field(..., description="The provider to use")
    api_key: SecretStr = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Optional custom base URL for the provider")
    blacklist_models: Optional[list[str]] = Field(None, description="models selection for blacklist")

class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field("ok", description="The message of the operation")