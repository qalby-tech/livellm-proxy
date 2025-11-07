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

    async def text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> bytes:
        config = gen_config or {}
        
        with logfire.span(
            f"text2speech {model}",
            _span_name=f"text2speech {model}",
            **{
                "gen_ai.provider.name": "elevenlabs",
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": model,
                "gen_ai.request.voice": voice,
                "gen_ai.request.text_length": len(text),
                "gen_ai.request.text_content": text,
                "gen_ai.request.output_format": self.default_output_format,
                "gen_ai.request.sample_rate": self.default_sample_rate,
                "gen_ai.request.parameters": config
            }
        ) as span:
            speech = await self.client.text_to_speech.convert(
                text=text,
                voice_id=voice,
                model_id=model,
                output_format=self.default_output_format,
                **config
            )
            # Collect all chunks efficiently
            chunks = [chunk for chunk in speech]
            audio_bytes = b"".join(chunks)
            
            span.set_attribute("gen_ai.response.audio_size_bytes", len(audio_bytes))
            span.set_attribute("gen_ai.response.duration_seconds", len(audio_bytes) / (self.default_sample_rate * 2))
            
            return audio_bytes
    
    
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        config = gen_config or {}
        
        with logfire.span(
            f"stream_text2speech {model}",
            _span_name=f"stream_text2speech {model}",
            **{
                "gen_ai.provider.name": "elevenlabs",
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": model,
                "gen_ai.request.voice": voice,
                "gen_ai.request.text_length": len(text),
                "gen_ai.request.text_content": text,
                "gen_ai.request.output_format": self.default_output_format,
                "gen_ai.request.sample_rate": self.default_sample_rate,
                "gen_ai.request.parameters": config
            }
        ) as span:
            total_bytes = 0
            async for chunk in self.client.text_to_speech.stream(
                text=text,
                voice_id=voice,
                model_id=model,
                output_format=self.default_output_format,
                **config
            ):
                total_bytes += len(chunk)
                yield chunk
            
            span.set_attribute("gen_ai.response.audio_size_bytes", total_bytes)
            span.set_attribute("gen_ai.response.duration_seconds", total_bytes / (self.default_sample_rate * 2))
 
   
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        filename, file_content, content_type = request.file
        
        with logfire.span(
            f"transcribe {request.model}",
            _span_name=f"transcribe {request.model}",
            **{
                "gen_ai.provider.name": "elevenlabs",
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": request.model,
                "gen_ai.request.language": request.language or "auto",
                "gen_ai.request.filename": filename,
                "gen_ai.request.content_type": content_type,
                "gen_ai.request.file_size_bytes": len(file_content),
                "gen_ai.request.parameters": config
            }
        ) as span:
            transcription: SpeechToTextConvertResponse = await self.client.speech_to_text.convert(
                file=request.file,
                language_code=request.language,
                model_id=request.model,
                **config
            )
            
            response = TranscribeResponse(
                text=transcription.text, 
                language=transcription.language_code
            )
            
            span.set_attribute("gen_ai.response.text_length", len(response.text))
            span.set_attribute("gen_ai.response.text_preview", response.text[:200] if len(response.text) > 200 else response.text)
            span.set_attribute("gen_ai.response.language", response.language)
            
            return response
