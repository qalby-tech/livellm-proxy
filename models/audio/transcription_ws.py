from pydantic import BaseModel, Field


class TranscriptionInitWsRequest(BaseModel):
    provider_uid: str = Field(..., description="The provider uid")
    model: str = Field(..., description="The model")
    language: str = Field(default="auto", description="The language")
    input_sample_rate: int = Field(default=24000, description="The input sample rate")
    gen_config: dict = Field(default={}, description="The generation configuration")


class TranscriptionAudioChunkWsRequest(BaseModel):
    audio: str = Field(..., description="The audio (base64 encoded)")

class TranscriptionWsResponse(BaseModel):
    transcription: str = Field(..., description="The transcription")
    is_end: bool = Field(..., description="Whether the response is the end of the transcription")
