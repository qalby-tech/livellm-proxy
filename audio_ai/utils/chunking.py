from typing import AsyncGenerator, AsyncIterator


class ChunkCollector:
    """
    Implementation for collecting audio chunks and yielding exactly chunk_size_ms audio bytes
    """
    
    def __init__(self, sample_rate: int, chunk_size_ms: int = 20):
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        # Calculate exact bytes needed for chunk_size_ms
        # For PCM16: 2 bytes per sample (16-bit audio)
        self.bytes_per_chunk = int((self.chunk_size_ms / 1000.0) * self.sample_rate * 2)
        self.audio_buffer = b""
    
    async def collect_chunks(self, audio_data: bytes) -> AsyncGenerator[bytes, None]:
        """
        Add audio data to buffer and yield exactly chunk_size_ms chunks
        """
        self.audio_buffer += audio_data
        
        # Yield complete chunks of exactly chunk_size_ms
        while len(self.audio_buffer) >= self.bytes_per_chunk:
            chunk = self.audio_buffer[:self.bytes_per_chunk]
            self.audio_buffer = self.audio_buffer[self.bytes_per_chunk:]
            yield chunk
    
    def flush(self) -> bytes:
        """
        Flush any remaining audio data in the buffer
        """
        remaining = self.audio_buffer
        self.audio_buffer = b""
        return remaining

    async def process_stream(self, pcm16_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for pcm16 in pcm16_stream:
            # Collect chunks of exactly chunk_size_ms
            async for chunk in self.collect_chunks(pcm16):
                yield chunk
        
        # Flush any remaining audio data at the end of the stream
        remaining = self.flush()
        if remaining:
            yield remaining
    
    async def process_chunk(self, pcm16: bytes) -> AsyncIterator[bytes]:
        return self.collect_chunks(pcm16)
