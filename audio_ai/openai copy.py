import asyncio
from openai import AsyncOpenAI
from openai import HttpxBinaryResponseContent
from openai.types.audio import Transcription
from audio_ai.base import AudioAIService
from audio_ai.base import AudioRealtimeTranscriptionService
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from typing import Optional, AsyncIterator
import websockets
from models.audio.openai_ws import OpenaiWSTranscriptionDelta, OpenaiWSTranscriptionEnd, OpenaiWSTranscriptionResponse, OpenaiWsResponse
from models.audio.transcription_ws import TranscriptionWsResponse, TranscriptionAudioChunkWsRequest
import json
import base64
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
            speech: HttpxBinaryResponseContent = await self.client.audio.speech.create(
                model=model,
                input=text,
                voice=voice,
                response_format=self.default_output_format,
                **config
            )
            audio_bytes = speech.content
            span.set_attribute("gen_ai.response.audio_size_bytes", len(audio_bytes))
            span.set_attribute("gen_ai.response.duration_seconds", len(audio_bytes) / (self.default_sample_rate * 2))
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
            total_bytes = 0
            async with self.client.audio.speech.with_streaming_response.create(
                model=model,
                input=text,
                voice=voice,
                response_format=self.default_output_format,
                **config
            ) as stream_response:
                async for chunk in stream_response.iter_bytes():
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
        api_key: str, 
        model: str = "gpt-4o-mini-transcribe", 
        language: str = "auto", 
        input_sample_rate: int = 24000,
        base_url: str = "wss://api.openai.com/v1"
        ):
        if base_url.startswith("http"):
            base_url = base_url.replace("http", "ws")
        if base_url.startswith("https"):
            base_url = base_url.replace("https", "wss")
        self.base_url = f"{base_url.rstrip('/')}/realtime"
        self.api_key = api_key
        self.url = f"{self.base_url}?intent=transcription"
        self.language = language
        self.model = model
        
        # configs
        self.turn_threshold = 0.5
        self.turn_prefix_padding_ms = 100
        self.turn_silence_duration_ms = 100
        self.noise_reduction_type = "near_field" # far_field or null
        self.input_sample_rate = input_sample_rate

        # session
        self.session = None

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
    
    
    @property
    def transcription_session(self) -> dict:
        return {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": self.model, "language": self.language},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.turn_threshold,
                    "prefix_padding_ms": self.turn_prefix_padding_ms,
                    "silence_duration_ms": self.turn_silence_duration_ms,
                },
                "input_audio_noise_reduction": self.noise_reduction_type,
            },
        }
    

    async def connect(self) -> None:
        logfire.info(f"Connecting to OpenAI WebSocket at {self.url}")
        self.session = await websockets.connect(self.url, additional_headers=self.headers)
        logfire.info(f"Connected to OpenAI WebSocket")
    
    async def send_audio_chunk(self, audio_source: AsyncIterator[TranscriptionAudioChunkWsRequest]) -> None:
        async for chunk in audio_source:
            await self.session.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": chunk.audio
                    }
                )
            )
    
    async def recieve_audio_chunk(self, audio_sink: asyncio.Queue[TranscriptionWsResponse]) -> None:
        """
        Continuously receive transcription chunks from OpenAI WebSocket and put them in the queue
        """
        try:
            async for response in self.session:
                message: dict = json.loads(response)
                message_type = message.get("type", "Unknown")
                
                # Log for debugging
                print(message)
                print(message_type)
                
                match message_type:
                    case "conversation.item.input_audio_transcription.delta":
                        delta = OpenaiWSTranscriptionDelta.model_validate(message)
                        await audio_sink.put(TranscriptionWsResponse(
                            transcription=delta.delta,
                            is_end=False
                        ))
                    case "conversation.item.input_audio_transcription.completed":
                        completion = OpenaiWSTranscriptionEnd.model_validate(message)
                        await audio_sink.put(TranscriptionWsResponse(
                            transcription=completion.transcription,
                            is_end=True
                        ))
                    case "session.created":
                        await self.session.send(
                            json.dumps(
                                self.transcription_session
                            )
                        )
                    case "session.updated" | "input_audio_buffer.speech_started" | "input_audio_buffer.speech_stopped" | "input_audio_buffer.committed":
                        # These are status messages we can ignore or log
                        logfire.debug(f"Status message: {message_type}")
                        continue
                    case "error":
                        error_msg = message.get("error", {}).get("message", "Unknown error")
                        logfire.error(f"OpenAI WebSocket error: {error_msg}")
                        raise ValueError(f"OpenAI error: {error_msg}")
                    case _:
                        logfire.warning(f"Unknown message type: {message_type}, message: {message}")
                        print(message)
        except websockets.exceptions.ConnectionClosed:
            logfire.info("WebSocket connection closed")
        except Exception as e:
            logfire.error(f"Error receiving audio chunk: {e}", exc_info=True)
            raise
    
    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
            


    # async def main(self):
    #     client = AsyncOpenAI(api_key=self.api_key)
    #     await client.beta.realtime.transcription_sessions.create()