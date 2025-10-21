from pydantic import BaseModel, Field
from typing import Optional
from models.audio.audio import AudioSettings


class SpeakRequest(BaseModel):
    settings: AudioSettings = Field(..., description="The settings for the audio")
    text: str = Field(..., description="The text to speak")
    voice: str = Field(..., description="The voice to use")
    output_format: str = Field(..., description="The output format of the audio")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")


class SpeakResponse(BaseModel):
    audio: bytes = Field(..., description="The audio data")
    content_type: str = Field(..., description="The content type of the audio")
    sample_rate: int = Field(..., description="The sample rate of the output audio")
