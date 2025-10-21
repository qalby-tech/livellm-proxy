from pydantic import BaseModel, Field
from models.audio.audio import AudioSettings
from typing import Tuple, TypeAlias, Optional
from models.audio.audio import AudioUsage

File: TypeAlias = Tuple[str, bytes, str] # (filename, file_content, content_type)

class TranscribeRequest(BaseModel):
    settings: AudioSettings = Field(..., description="The settings for the transcription")
    file: File = Field(..., description="The file to transcribe")
    language: Optional[str] = Field(default=None, description="The language to transcribe")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="The text of the transcription")
    language: Optional[str] = Field(default=None, description="The language of the transcription")
    usage: AudioUsage = Field(..., description="The usage of the transcription")
