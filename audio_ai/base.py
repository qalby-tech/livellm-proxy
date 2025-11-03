from abc import ABC, abstractmethod
from typing import Tuple, AsyncIterator, Optional
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from audio_ai.utils import ChunkCollector, Resampler

class AudioAIService(ABC):
    mime_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "ulaw": "audio/ulaw",
        "alaw": "audio/alaw",
        "pcm": "audio/pcm"
    }
    pcm_mime_type = "audio/pcm"

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
        audio = await resampler.process_chunk(audio)
        return SpeakResponse(audio=audio, content_type=self.pcm_mime_type, sample_rate=self.default_sample_rate)

    
    async def stream_speak(self, request: SpeakRequest) -> SpeakStreamResponse:
        """
        Returns a tuple of (async iterator of bytes, mime type, sample rate)
        """
        generator = await self.stream_text2speech(request.model, request.text, request.voice, request.gen_config)
        first_chunk = await generator.__anext__()
        resampler = Resampler(self.default_sample_rate, request.sample_rate)
        chunk_collector = ChunkCollector(request.sample_rate, request.chunk_size)
        async def _generator(first_chunk: bytes, generator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
            yield first_chunk
            async for chunk in generator:
                yield chunk
        native_generator = _generator(first_chunk, generator)
        chunked_generator = chunk_collector.process_stream(native_generator)
        resampled_generator = resampler.process_stream(chunked_generator)
        return resampled_generator, self.pcm_mime_type, self.default_sample_rate

    
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """
        Transcribe audio to text
        """
        pass