import asyncio
import wave
import io
from audioop import lin2ulaw, lin2alaw, ulaw2lin, alaw2lin
from models.audio.speak import SpeakMimeType
from typing import AsyncIterator


def pcm2wav(pcm16: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    buffer.seek(0)
    audio = buffer.getvalue()
    buffer.close()
    return audio


async def encode(audio: bytes, mime_type: SpeakMimeType) -> bytes:
    """
    transforms pcm chunks to the target mime type
    """
    if mime_type == SpeakMimeType.PCM:
        return audio
    elif mime_type == SpeakMimeType.ULAW:
        return lin2ulaw(audio, 2)
    elif mime_type == SpeakMimeType.ALAW:
        return lin2alaw(audio, 2)
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")


async def decode(audio: bytes, mime_type: SpeakMimeType) -> bytes:
    """
    transforms the input audio chunks to pcm chunks
    """
    if mime_type == SpeakMimeType.PCM:
        return audio
    elif mime_type == SpeakMimeType.ULAW:
        return ulaw2lin(audio, 2)
    elif mime_type == SpeakMimeType.ALAW:
        return alaw2lin(audio, 2)
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

async def encode_from_pcm_stream(generator: AsyncIterator[bytes], mime_type: SpeakMimeType) -> AsyncIterator[bytes]:
    """
    Transform a stream of PCM chunks into the target mime type.

    This is implemented as an async *generator* so it can be passed
    directly to FastAPI/Starlette `StreamingResponse`.
    """
    if mime_type == SpeakMimeType.WAV or mime_type == SpeakMimeType.MP3:
        # These formats are not supported in streaming mode because they
        # typically require full-file headers.
        raise ValueError(f"{mime_type.value} encoding is not supported")

    async for audio in generator:
        # Yield each encoded chunk instead of returning once.
        yield await encode(audio, mime_type)


async def decode_into_pcm_stream(generator: AsyncIterator[bytes], mime_type: SpeakMimeType) -> AsyncIterator[bytes]:
    """
    transforms the input audio chunks to pcm chunks
    """
    async for audio in generator:
        yield await decode(audio, mime_type)
