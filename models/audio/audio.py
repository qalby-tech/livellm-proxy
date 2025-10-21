from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AudioProviderKind(Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"


class AudioUsage(BaseModel):
    input_tokens: int = Field(default=0, description="The number of input tokens")
    output_tokens: int = Field(default=0, description="The number of output tokens")

class AudioSettings(BaseModel):
    provider: AudioProviderKind
    api_key: str
    model: str
    base_url: Optional[str] = Field(None, description="The base URL for the provider")


