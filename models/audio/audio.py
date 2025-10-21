from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AudioProviderKind(Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"


class AudioSettings(BaseModel):
    provider: AudioProviderKind
    api_key: str
    model: str
    base_url: Optional[str] = Field(None, description="The base URL for the provider")


