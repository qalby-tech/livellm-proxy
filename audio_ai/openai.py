from openai import AsyncOpenAI
from openai import HttpxBinaryResponseContent
from openai.types.audio import Transcription
from audio_ai.base import AudioAIService
from models.common import Settings
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from typing import Optional, Tuple, AsyncIterator
import logfire

class OpenAIAudioAIService(AudioAIService):
    default_output_format = "pcm"

    def __init__(self, client: AsyncOpenAI):
        self.client = client
    

    @property
    def default_sample_rate(self) -> int:
        return 24000
    

    async def text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> bytes:
        config = gen_config or {}
        speech: HttpxBinaryResponseContent = await self.client.audio.speech.create(
            model=model,
            input=text,
            voice=voice,
            response_format=self.default_output_format,
            **config
        )
        return speech.content
    
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        config = gen_config or {}
        async with self.client.audio.speech.with_streaming_response.create(
            model=model,
            input=text,
            voice=voice,
            response_format=self.default_output_format,
            **config
        ) as stream_response:
            async for chunk in stream_response.iter_bytes():
                yield chunk
    
    @logfire.instrument(span_name="OpenAI Transcribe", record_return=True)
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        transcription: Transcription = await self.client.audio.transcriptions.create(
            model=request.model,
            file=request.file,
            language=request.language,
            **config
        )
        return TranscribeResponse(
            text=transcription.text, 
            language=request.language or "auto"
            )
    