from pydantic import BaseModel, Field
from typing import Optional, TypeAlias, Tuple, AsyncIterator


SpeakStreamResponse: TypeAlias = Tuple[AsyncIterator[bytes], str, int]

class SpeakRequest(BaseModel):
    model: str = Field(..., description="The model to use")
    text: str = Field(..., description="The text to speak")
    voice: str = Field(..., description="The voice to use")
    mime_type: str = Field(..., description="The MIME type of the output audio (e.g., audio/pcm, audio/wav, audio/mpeg)")
    sample_rate: int = Field(..., description="The target sample rate of the output audio")
    chunk_size: int = Field(default=20, description="Chunk size in milliseconds for streaming (default: 20ms)")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")


class SpeakResponse(BaseModel):
    audio: bytes = Field(..., description="The audio data")
    content_type: str = Field(..., description="The content type of the audio")
    sample_rate: int = Field(..., description="The sample rate of the output audio")
