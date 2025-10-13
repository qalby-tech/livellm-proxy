from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str = Field(description="The text of the transcription")


class SpeakResponse(BaseModel):
    audio_base64: str = Field(description="The audio of the speech")

class ChatResponse(BaseModel):
    text: str = Field(description="The text of the run")