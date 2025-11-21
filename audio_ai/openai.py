import asyncio
from openai import AsyncOpenAI
from openai import HttpxBinaryResponseContent
from openai.types.audio import Transcription
from audio_ai.base import AudioAIService
from audio_ai.base import AudioRealtimeTranscriptionService
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from typing import Optional, AsyncIterator, Awaitable, Callable
from models.audio.transcription_ws import TranscriptionWsResponse
import numpy as np
from agents.voice.models.openai_stt import OpenAISTTModel, STTModelSettings
from agents.voice.input import StreamedAudioInput
from agents.voice.model import StreamedTranscriptionSession
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
        
        with logfire.span(
            f"text2speech {model}",
            _span_name=f"text2speech {model}",
            **{
                "gen_ai.provider.name": "openai",
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
            pass
        speech: HttpxBinaryResponseContent = await self.client.audio.speech.create(
            model=model,
            input=text,
            voice=voice,
            response_format=self.default_output_format,
            **config
        )
        audio_bytes = speech.content
        return audio_bytes
    
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        config = gen_config or {}
        
        with logfire.span(
            f"stream_text2speech {model}",
            _span_name=f"stream_text2speech {model}",
            **{
                "gen_ai.provider.name": "openai",
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
            pass
        async with self.client.audio.speech.with_streaming_response.create(
            model=model,
            input=text,
            voice=voice,
            response_format=self.default_output_format,
            **config
        ) as stream_response:
            async for chunk in stream_response.iter_bytes():
                yield chunk
    
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        filename, file_content, content_type = request.file
        
        with logfire.span(
            f"transcribe {request.model}",
            _span_name=f"transcribe {request.model}",
            **{
                "gen_ai.provider.name": "openai",
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": request.model,
                "gen_ai.request.language": request.language or "auto",
                "gen_ai.request.filename": filename,
                "gen_ai.request.content_type": content_type,
                "gen_ai.request.file_size_bytes": len(file_content),
                "gen_ai.request.parameters": config
            }
        ) as span:
            transcription: Transcription = await self.client.audio.transcriptions.create(
                model=request.model,
                file=request.file,
                language=request.language,
                **config
            )
            
            response = TranscribeResponse(
                text=transcription.text, 
                language=request.language or "auto"
            )
            
            span.set_attribute("gen_ai.response.text_length", len(response.text))
            span.set_attribute("gen_ai.response.text_preview", response.text[:200] if len(response.text) > 200 else response.text)
            span.set_attribute("gen_ai.response.language", response.language)
            
            return response


class OpenAIRealtimeTranscriptionService(AudioRealtimeTranscriptionService):

    def __init__(
        self, 
        model: str, 
        openai_client: AsyncOpenAI,
        language: str = "auto",
        gen_config: Optional[dict] = None
        ):
        self.model = OpenAISTTModel(model, openai_client)
        
        # configs
        gen_config = gen_config or {}
        self.language = language
        self.turn_threshold = gen_config.get("turn_threshold", 0.5)
        self.turn_prefix_padding_ms = gen_config.get("turn_prefix_padding_ms", 100)
        self.turn_silence_duration_ms = gen_config.get("turn_silence_duration_ms", 100)
        self.noise_reduction_type = gen_config.get("noise_reduction_type", "near_field") # far_field or null
        self.prompt = gen_config.get("prompt", "")

        # internal
        self.__input = StreamedAudioInput() # then use add_audio
        self.__session: Optional[StreamedTranscriptionSession] = None


    @property
    def default_sample_rate(self) -> int:
        return 24000
   
    @property
    def stt_settings(self) -> STTModelSettings:
        return STTModelSettings(
            prompt=self.prompt,
            language=self.language,
            turn_detection={
                "type": "server_vad",
                "threshold": self.turn_threshold,
                "prefix_padding_ms": self.turn_prefix_padding_ms,
                "silence_duration_ms": self.turn_silence_duration_ms,
        }
        )

    async def connect(self) -> None:
        self.__session = await self.model.create_session(
            input=self.__input,
            settings=self.stt_settings,
            trace_include_sensitive_audio_data=False,
            trace_include_sensitive_data=False
        )
    
    async def send_audio_chunk(self, audio_source: AsyncIterator[bytes]) -> None:
        """
        Decode base64 audio chunks and add them to the input stream.
        Expects PCM16 audio data encoded as base64.
        """
        async for chunk in audio_source:
            try:                    
                # Convert bytes to NumPy array of int16 (PCM16 format)
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                
                # Add the audio array to the streamed input
                await self.__input.add_audio(audio_array) # accepts only pcm, 24_000 sample rate
            except (ValueError, TypeError) as e:
                logfire.error(f"Error decoding audio chunk: {e}", exc_info=True)
                # Continue processing other chunks even if one fails
                continue
    
    async def receive_audio_chunk(self, audio_sink: Callable[[str], Awaitable[None]]) -> None:
        """
        Continuously receive transcription chunks from OpenAI WebSocket and send them to the client
        params:
        - audio_sink: callable to send transcription responses to
        """
        async for transcription in self.__session.transcribe_turns():
            await audio_sink(transcription)
    
    async def disconnect(self) -> None:
        if self.__session:
            try:
                await self.__session.close()
            except Exception as e:
                logfire.error(f"Error closing session: {e}", exc_info=True)
            finally:
                self.__session = None
