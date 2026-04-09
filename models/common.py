"""Common models shared across all services"""

from pydantic import BaseModel, Field, SecretStr
from enum import Enum
from typing import Optional, Dict


class BaseRequest(BaseModel):
    """Base request model that all service requests inherit from"""
    provider_uid: str = Field(..., description="The unique identifier of the provider configuration to use")


class ProviderKind(str, Enum):
    """Unified provider types for both agent and audio services"""
    # Agent providers
    OPENAI = "openai"
    OPENAI_CHAT = "openai_chat" # /v1/completions api, legacy
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    # Audio providers
    ELEVENLABS = "elevenlabs"


class ContextOverflowStrategyType(str, Enum):
    """Strategy for handling context overflow when text exceeds context_limit."""
    TRUNCATE = "truncate"  # Take beginning, middle, and end portions
    RECYCLE = "recycle"    # Iteratively process chunks, merging results


class FallbackStrategyType(str, Enum):
    """Strategy for executing fallbacks"""
    SEQUENTIAL = "sequential"  # Try primary first, then fallback on failure
    PARALLEL = "parallel"      # Try primary and fallback simultaneously, take first success


class FallbackConfig(BaseModel):
    """Fallback configuration for a specific model"""
    fallback_provider_uid: str = Field(..., description="The provider UID to fallback to")
    fallback_model: str = Field(..., description="The model to use on the fallback provider")
    fallback_strategy: FallbackStrategyType = Field(
        default=FallbackStrategyType.SEQUENTIAL,
        description="Strategy for fallback: 'sequential' (try primary then fallback) or 'parallel' (try both simultaneously)"
    )
    context_limit: int = Field(default=0, description="Maximum context size in tokens for the fallback model. If <= 0, context is assumed not to be an issue.")
    context_overflow_strategy: ContextOverflowStrategyType = Field(
        default=ContextOverflowStrategyType.TRUNCATE,
        description="Strategy for handling context overflow on the fallback model: 'truncate' or 'recycle'"
    )


class ModelConfig(BaseModel):
    """Per-model configuration including fallback and context settings"""
    fallback: Optional[FallbackConfig] = Field(None, description="Fallback configuration for this model")
    context_limit: int = Field(default=0, description="Maximum context size in tokens. If <= 0, context is assumed not to be an issue.")
    context_overflow_strategy: ContextOverflowStrategyType = Field(
        default=ContextOverflowStrategyType.TRUNCATE,
        description="Strategy for handling context overflow: 'truncate' or 'recycle'"
    )


class Settings(BaseModel):
    """Base settings for all service providers"""
    uid: str = Field(..., description="The unique identifier of the provider configuration")
    provider: ProviderKind = Field(..., description="The provider to use")
    api_key: SecretStr = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Optional custom base URL for the provider")
    blacklist_models: Optional[list[str]] = Field(None, description="models selection for blacklist")
    model_configs: Optional[Dict[str, ModelConfig]] = Field(
        None, 
        description="Per-model configuration including fallback and context settings. Key is model name."
    )


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field("ok", description="The message of the operation")
