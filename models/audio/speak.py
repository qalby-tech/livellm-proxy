from pydantic import BaseModel, Field, field_validator
from typing import Optional, TypeAlias, Tuple, AsyncIterator
from enum import Enum
from models.common import BaseRequest

SpeakStreamResponse: TypeAlias = Tuple[AsyncIterator[bytes], str, int]


class SpeakMimeType(Enum):
    PCM = "audio/pcm"
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    ULAW = "audio/ulaw"
    ALAW = "audio/alaw"

class SpeakRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    text: str = Field(..., description="The text to speak")
    voice: str = Field(..., description="The voice to use")
    mime_type: SpeakMimeType = Field(..., description="The MIME type of the output audio")
    sample_rate: int = Field(..., description="The target sample rate of the output audio")
    chunk_size: int = Field(default=20, description="Chunk size in milliseconds for streaming (default: 20ms)")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v < 8000 or v > 48000:
            raise ValueError("Sample rate must be between 8000 and 48000 Hz")
        return v
    
    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        if v > 1000:
            raise ValueError("Chunk size must not exceed 1000ms")
        return v


class SpeakResponse(BaseModel):
    audio: bytes = Field(..., description="The audio data")
    content_type: str = Field(..., description="The content type of the audio")
    sample_rate: int = Field(..., description="The sample rate of the output audio")
