from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse, SpeakMimeType
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from audio_ai.utils import ChunkCollector, Resampler
from audio_ai.utils import AudioEncoder


class AudioAIService(ABC):

    pcm_mime_type = SpeakMimeType.PCM.value

    @abstractmethod
    @property
    def default_sample_rate(self) -> int:
        return 24000
    
    @abstractmethod
    async def text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> bytes:
        pass

    @abstractmethod
    async def stream_text2speech(self, model: str, text: str, voice: str, gen_config: Optional[dict] = None) -> AsyncIterator[bytes]:
        pass
  
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        audio = await self.text2speech(request.model, request.text, request.voice, request.gen_config)
        resampler = Resampler(self.default_sample_rate, request.sample_rate)
        audio = await resampler.process_chunk(audio, flush=True)
        encoder = AudioEncoder(request.sample_rate, request.mime_type)
        audio = await encoder.encode(audio)
        return SpeakResponse(audio=audio, content_type=request.mime_type.value, sample_rate=request.sample_rate)

    
    async def stream_speak(self, request: SpeakRequest) -> SpeakStreamResponse:
        """
        Returns a tuple of (async iterator of bytes, mime type, sample rate)
        Processing pipeline: native audio -> resample -> chunk to fixed sizes -> encode
        """
        generator = await self.stream_text2speech(request.model, request.text, request.voice, request.gen_config)
        first_chunk = await generator.__anext__()
        resampler = Resampler(self.default_sample_rate, request.sample_rate)
        chunk_collector = ChunkCollector(request.sample_rate, request.chunk_size)
        encoder = AudioEncoder(request.sample_rate, request.mime_type)
        async def _generator(first_chunk: bytes, generator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
            yield first_chunk
            async for chunk in generator:
                yield chunk
        native_generator = _generator(first_chunk, generator)
        # IMPORTANT: Resample first, then chunk (chunking needs correct sample rate), then encode
        resampled_generator = resampler.process_stream(native_generator)
        chunked_generator = chunk_collector.process_stream(resampled_generator)
        encoded_generator = encoder.encode_stream(chunked_generator)
        return encoded_generator, request.mime_type.value, request.sample_rate

    
    @abstractmethod
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """
        Transcribe audio to text
        """
        pass