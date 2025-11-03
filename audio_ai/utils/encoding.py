import asyncio
import wave
import io
from audioop import lin2ulaw, lin2alaw
from models.audio.speak import SpeakMimeType
from typing import AsyncIterator, Callable, Optional


class AudioEncoder:

    def __init__(self, sample_rate: int, mime_type: SpeakMimeType):
        self.sample_rate = sample_rate
        self.mime_type = mime_type
        self.encoder: Optional[Callable[[bytes], bytes]] = None
        self.init_encoder()

    def _identity(self, data: bytes) -> bytes:
        return data

    def init_encoder(self):
        match self.mime_type:
            case SpeakMimeType.PCM:
                self.encoder = self._identity
            case SpeakMimeType.WAV:
                self.encoder = self.pcm2wav
            case SpeakMimeType.ULAW:
                self.encoder = self.pcm2ulaw
            case SpeakMimeType.ALAW:
                self.encoder = self.pcm2alaw
            case SpeakMimeType.MP3:
                raise NotImplementedError("MP3 encoding is not yet implemented")
            case _:
                raise ValueError(f"Unsupported MIME type: {self.mime_type}")

    def pcm2wav(self, pcm16: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm16)
        buffer.seek(0)
        audio = buffer.getvalue()
        buffer.close()
        return audio

    def pcm2ulaw(self, pcm16: bytes) -> bytes:
        return lin2ulaw(pcm16, 2)

    def pcm2alaw(self, pcm16: bytes) -> bytes:
        return lin2alaw(pcm16, 2)

    async def encode(self, audio: bytes) -> bytes:
        return await asyncio.to_thread(self.encoder, audio)

    async def encode_stream(self, generator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        if self.mime_type == SpeakMimeType.WAV \
            or self.mime_type == SpeakMimeType.MP3:
            raise ValueError(f"{self.mime_type.value} encoding is not supported for streaming")
        async for audio in generator:
            yield await self.encode(audio)
    
