import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Awaitable, Callable
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse, SpeakMimeType
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.audio.transcription_ws import TranscriptionWsResponse
from audio_ai.utils import ChunkCollector, Resampler
from audio_ai.utils.encoding import encode, decode, encode_from_pcm_stream, decode_into_pcm_stream
import logfire
import time


class AudioAIService(ABC):

    pcm_mime_type = SpeakMimeType.PCM.value

    @property
    @abstractmethod
    def default_sample_rate(self) -> int:
        return 24000
    
    @abstractmethod
    async def text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> bytes:
        pass

    @abstractmethod
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        pass
  
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        with logfire.span(
            "text2speech",
            model=request.model,
            voice=request.voice,
            provider_uid=request.provider_uid,
            text_length=len(request.text),
            target_sample_rate=request.sample_rate,
            target_mime_type=request.mime_type.value,
            source_sample_rate=self.default_sample_rate
        ) as span:
            start_time = time.time()
            audio = await self.text2speech(request.model, request.text, request.voice, request.gen_config)
            time_to_first_token = time.time() - start_time
            span.set_attribute("gen_ai.server.time_to_first_token", time_to_first_token)
            span.set_attribute("raw_audio_size_bytes", len(audio))
            
            resampler = Resampler(self.default_sample_rate, request.sample_rate)
            audio = await resampler.process_chunk(audio, flush=True)
            span.set_attribute("resampled_audio_size_bytes", len(audio))
            
            audio = await encode(audio, request.mime_type)
            span.set_attribute("encoded_audio_size_bytes", len(audio))
            
            return SpeakResponse(audio=audio, content_type=request.mime_type.value, sample_rate=request.sample_rate)

    
    async def stream_speak(self, request: SpeakRequest) -> SpeakStreamResponse:
        """
        Returns a tuple of (async iterator of bytes, mime type, sample rate)
        Processing pipeline: native audio -> resample -> chunk to fixed sizes -> encode
        """
        try:
            with logfire.span(
                "streamng text2speech",
                model=request.model,
                voice=request.voice,
                provider_uid=request.provider_uid,
                text_length=len(request.text),
                target_sample_rate=request.sample_rate,
                target_mime_type=request.mime_type.value,
                chunk_size_ms=request.chunk_size,
                source_sample_rate=self.default_sample_rate
            ) as span:
                start_time = time.time()
                generator = self.stream_text2speech(request.model, request.text, request.voice, request.gen_config)
                
                # Handle empty generator case
                first_chunk = await generator.__anext__()
                time_to_first_token = time.time() - start_time
                span.set_attribute("gen_ai.server.time_to_first_token", time_to_first_token)
                                
            resampler = Resampler(self.default_sample_rate, request.sample_rate)
            chunk_collector = ChunkCollector(request.sample_rate, request.chunk_size)
            
            async def _generator(first_chunk: bytes, generator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
                yield first_chunk
                async for chunk in generator:
                    yield chunk
            native_generator = _generator(first_chunk, generator)
            # Resample, chunk, encode
            resampled_generator = resampler.process_stream(native_generator)
            chunked_generator = chunk_collector.process_stream(resampled_generator)
            
            # Encode stream            
            encoded_generator = encode_from_pcm_stream(chunked_generator, request.mime_type)
            return encoded_generator, request.mime_type.value, request.sample_rate
        except StopAsyncIteration:
            async def empty_generator() -> AsyncIterator[bytes]:
                yield b""
            return empty_generator(), request.mime_type.value, request.sample_rate


    
    @abstractmethod
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """
        Transcribe audio to text
        """
        pass



class AudioRealtimeTranscriptionService(ABC):
    """
    wrapper for websocket connection for realtime transcription
    """
    pcm_mime_type = SpeakMimeType.PCM.value

    @property
    @abstractmethod
    def default_sample_rate(self) -> int:
        return 24000
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the websocket connection
        """
        pass
    
    
    @abstractmethod
    async def send_audio_chunk(self, audio_source: AsyncIterator[bytes]) -> None:
        """
        Send audio chunks to the transcription service
        params:
        - audio_source: async iterator of audio chunks (pcm16)
        """
        pass

    @abstractmethod
    async def receive_audio_chunk(self, audio_sink: Callable[[TranscriptionWsResponse], Awaitable[None]]) -> None:
        """
        Receive transcription chunks from the service and send them to the client
        params:
        - audio_sink: awaitable to send transcription responses to
        """
        pass


    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the websocket connection
        """
        pass


    async def realtime_transcribe(
        self, 
        audio_source: AsyncIterator[bytes], 
        audio_sink: Callable[[str], Awaitable[None]],
        input_audio_format: SpeakMimeType = SpeakMimeType.PCM,
        input_sample_rate: int = 24000
        ) -> None:
        """
        Transcribe audio to text in realtime
        params:
        - audio_source: async iterator yielding audio bytes
        - audio_sink: awaitable to send transcription responses to
        - input_audio_format: format of input audio (PCM, ulaw, alaw)
        - input_sample_rate: sample rate of input audio
        """
        send_task = None
        receive_task = None
        try:
            # preprate the audio source
            ## Decode from the input format (ulaw/alaw/pcm) to PCM
            decoded_audio_source = decode_into_pcm_stream(audio_source, input_audio_format)
            ## Resample to the service's required sample rate (typically 24kHz)
            resampled_audio_source = Resampler(input_sample_rate, self.default_sample_rate).process_stream(decoded_audio_source)
            ## create send task
            send_task = asyncio.create_task(self.send_audio_chunk(resampled_audio_source))

            # preprate the audio sink
            ## preprate the receiver
            receive_task = asyncio.create_task(self.receive_audio_chunk(audio_sink))
            
            # Wait for both tasks
            await asyncio.gather(send_task, receive_task)
        except Exception as e:
            # Cancel both tasks on error
            for task in [send_task, receive_task]:
                if task and not task.done():
                    task.cancel()
            raise