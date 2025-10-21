from openai import AsyncOpenAI
from openai import HttpxBinaryResponseContent
from openai.types.audio import TranscriptionCreateResponse
from openai.types.audio import Transcription
from audio_ai.base import AudioAIService
from models.audio.speak import SpeakRequest, SpeakResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.audio.audio import AudioUsage
from typing import Optional, Tuple
import logfire

class OpenAIAudioAIService(AudioAIService):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    def decode_output_format(self, output_format: str) -> Tuple[str, int]:
        """
        openai output matched with mime_type, + sample rate is always 24_000
        """
        mime_type = self.mime_type_map.get(output_format, None)
        if mime_type:
            return mime_type, 24000
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    @logfire.instrument(span_name="OpenAI Speak", record_return=True)
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        config = request.gen_config or {}
        mime_type, sample_rate = self.decode_output_format(request.output_format)
        speech: HttpxBinaryResponseContent = await self.client.audio.speech.create(
            model=request.settings.model,
            input=request.text,
            voice=request.voice,
            response_format=request.output_format,
            **config
        )
        return SpeakResponse(audio=speech.content, content_type=mime_type, sample_rate=sample_rate)
    
    @logfire.instrument(span_name="OpenAI Transcribe", record_return=True)
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        transcription: Transcription = await self.client.audio.transcriptions.create(
            model=request.settings.model,
            file=request.file,
            language=request.language,
            **config
        )
        return TranscribeResponse(
            text=transcription.text, 
            language=request.language or "auto", 
            usage=AudioUsage(
                input_tokens=0, 
                output_tokens=0
                )
            )