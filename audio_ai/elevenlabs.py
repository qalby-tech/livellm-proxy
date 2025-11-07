from elevenlabs import AsyncElevenLabs
from elevenlabs import SpeechToTextConvertResponse
from audio_ai.base import AudioAIService
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from typing import Tuple, AsyncIterator, Optional
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
import logfire


class ElevenLabsAudioAIService(AudioAIService):
    default_output_format = "pcm_s16le_16"
    sample_rate = 16000

    def __init__(self, client: AsyncElevenLabs):
        self.client = client
    
    @property
    def default_sample_rate(self) -> int:
        return self.sample_rate

    @logfire.instrument(span_name="ElevenLabs Text2Speech", record_return=True)
    async def text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> bytes:
        config = gen_config or {}
        speech = await self.client.text_to_speech.convert(
            text=text,
            voice_id=voice,
            model_id=model,
            output_format=self.default_output_format,
            **config
        )
        # Collect all chunks efficiently
        chunks = [chunk for chunk in speech]
        return b"".join(chunks)
    
    
    @logfire.instrument(span_name="ElevenLabs Stream Text2Speech", record_return=True)
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        config = gen_config or {}
        async for chunk in self.client.text_to_speech.stream(
            text=text,
            voice_id=voice,
            model_id=model,
            output_format=self.default_output_format,
            **config
        ):
            yield chunk
 
   
    @logfire.instrument(span_name="ElevenLabs Transcribe", record_return=True)
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        transcription: SpeechToTextConvertResponse = await self.client.speech_to_text.convert(
            file=request.file,
            language_code=request.language,
            model_id=request.model,
            **config
        )
        return TranscribeResponse(
            text=transcription.text, 
            language=transcription.language_code
        )
