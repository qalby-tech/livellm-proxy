from pydantic import BaseModel, Field
from typing import Tuple, TypeAlias, Optional

File: TypeAlias = Tuple[str, bytes, str] # (filename, file_content, content_type)

class TranscribeRequest(BaseModel):
    model: str = Field(..., description="The model to use")
    file: File = Field(..., description="The file to transcribe")
    language: Optional[str] = Field(default=None, description="The language to transcribe")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="The text of the transcription")
    language: Optional[str] = Field(default=None, description="The language of the transcription")
