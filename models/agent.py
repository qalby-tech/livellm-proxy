# models for agent settings

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class ProviderKind(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GROQ = "groq"

class AgentSettings(BaseModel):
    provider: ProviderKind
    api_key: str
    model: str
    base_url: Optional[str] = Field(None, description="The base URL for the provider")
