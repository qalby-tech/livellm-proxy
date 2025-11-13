from pydantic import BaseModel, Field



class OpenaiWsResponse(BaseModel):
    type: str = Field(..., description="The type of the response")

class OpenaiWSTranscriptionResponse(OpenaiWsResponse):
    event_id: str = Field(..., description="The event id")
    item_id: str = Field(..., description="The item id")
    content_index: int = Field(..., description="The content index")

class OpenaiWSTranscriptionDelta(OpenaiWSTranscriptionResponse):
    delta: str = Field(..., description="The delta of the transcription")

class OpenaiWSTranscriptionEnd(OpenaiWSTranscriptionResponse):
    transcription: str = Field(..., description="The transcription")

