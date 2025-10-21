from abc import ABC, abstractmethod
from typing import Tuple, AsyncIterator
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse

class AudioAIService(ABC):
    mime_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "ulaw": "audio/ulaw",
        "alaw": "audio/alaw",
        "pcm": "audio/pcm"
    }
    

    @abstractmethod
    def decode_output_format(self, output_format: str) -> Tuple[str, int]:
        """
        returns mime type and sample rate
        """
        pass
    
    @abstractmethod
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        pass

    @abstractmethod
    async def stream_speak(self, request: SpeakRequest) -> SpeakStreamResponse:
        """
        Returns a tuple of (async iterator of bytes, mime type, sample rate)
        """
        pass

    @abstractmethod
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """
        Transcribe audio to text
        """
        pass