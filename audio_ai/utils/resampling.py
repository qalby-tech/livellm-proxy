from math import gcd
from scipy.signal import resample_poly
import numpy as np
from typing import AsyncIterator, Optional
import logfire
import asyncio

# Constants for PCM16 audio processing
PCM16_MAX_VALUE = 32768.0  # 2^15 for 16-bit signed audio
PCM16_MAX_INT = 32767  # Maximum value for int16
RESAMPLER_TAIL_MULTIPLIER = 10  # Heuristic for filter tail length


def _resample_pcm16_sync(pcm16: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    """Synchronous resample pcm16 to target sample rate.
    pcm16 are even length bytes!
    This is the CPU-bound implementation that runs in a thread pool.
    """
    g = gcd(original_sample_rate, target_sample_rate) # greatest common divisor
    up = target_sample_rate // g # upsampling factor
    down = original_sample_rate // g # downsampling factor

    if not pcm16:
        return b''

    pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
    pcm16_array = pcm16_array.astype(np.float32) / PCM16_MAX_VALUE # convert and normalize to [-1, 1]
    out = resample_poly(
        pcm16_array, 
        up, 
        down, 
        padtype='line', 
        axis=0, 
        window=('kaiser', 14)
        )
    if isinstance(out, np.ndarray):
        # Handle NaN values by replacing them with zeros before conversion
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out_i16 = (out.clip(-1, 1) * PCM16_MAX_INT).astype(np.int16) # convert back to int16
        return out_i16.tobytes()
    else:
        logfire.warning("Resampled output is not a numpy array, returning empty bytes")
        return b''


async def resample_pcm16(pcm16: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    """Async wrapper for resampling that offloads CPU-bound work to thread pool."""
    return await asyncio.to_thread(_resample_pcm16_sync, pcm16, original_sample_rate, target_sample_rate)


class Resampler:

    def __init__(self, input_sample_rate: int = 24000, output_sample_rate: int = 16000):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_buffer = b""
        # Streaming resampler state
        g = gcd(self.input_sample_rate, self.output_sample_rate)
        self._up = self.output_sample_rate // g
        self._down = self.input_sample_rate // g
        # Keep an input-domain tail to preserve continuity across chunks
        # Heuristic tail length derived from polyphase filter length (similar to scipy default ~ 10*max(up,down))
        self._tail_len = RESAMPLER_TAIL_MULTIPLIER * max(self._up, self._down)
        self._input_tail_f32: Optional[np.ndarray] = None
    

    async def process_chunk(self, pcm16: bytes, flush: bool = False) -> bytes:
        try:
            # Ensure even number of bytes for int16 decoding
            if self.audio_buffer:
                pcm16 = self.audio_buffer + pcm16
                self.audio_buffer = b""

            if len(pcm16) % 2 != 0:
                if flush:
                    # When flushing, pad with zero byte to make it even
                    pcm16 = pcm16 + b'\x00'
                else:
                    self.audio_buffer = pcm16[-1:]
                    pcm16 = pcm16[:-1]

            if not pcm16:
                return b""

            if self.input_sample_rate == self.output_sample_rate:
                return pcm16

            # Stateful streaming resample to minimize boundary artifacts
            # Convert to float32 [-1, 1]
            curr_i16 = np.frombuffer(pcm16, dtype=np.int16)
            curr_f32 = curr_i16.astype(np.float32) / PCM16_MAX_VALUE

            if self._input_tail_f32 is not None and len(self._input_tail_f32) > 0:
                concat_in = np.concatenate([self._input_tail_f32, curr_f32])
                tail_in_len = len(self._input_tail_f32)
            else:
                concat_in = curr_f32
                tail_in_len = 0

            # Do chunked resample via the stateless helper
            resampled_bytes = await resample_pcm16(
                (concat_in.clip(-1.0, 1.0) * PCM16_MAX_INT).astype(np.int16).tobytes(),
                self.input_sample_rate,
                self.output_sample_rate,
            )

            out_i16_full = np.frombuffer(resampled_bytes, dtype=np.int16)
            # Compute and drop the portion corresponding to the prepended tail
            if tail_in_len > 0:
                tail_out_len = (tail_in_len * self._up) // self._down
                if tail_out_len > 0 and tail_out_len < len(out_i16_full):
                    out_i16 = out_i16_full[tail_out_len:]
                else:
                    out_i16 = out_i16_full
            else:
                out_i16 = out_i16_full

            # Update input tail for next call (keep last N input samples)
            if self._tail_len > 0:
                if len(concat_in) > self._tail_len:
                    self._input_tail_f32 = concat_in[-self._tail_len:]
                else:
                    self._input_tail_f32 = concat_in.copy()
            else:
                self._input_tail_f32 = None

            return out_i16.tobytes()
        except Exception as e:
            logfire.error(f"Resampling failed: {e}")
            raise
    
    async def process_stream(self, pcm16_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for pcm16 in pcm16_stream:
            result = await self.process_chunk(pcm16)
            if result:  # Only yield non-empty chunks
                yield result
        
        # Flush any remaining buffered data at the end of the stream
        if self.audio_buffer:
            final_chunk = await self.process_chunk(b'', flush=True)
            if final_chunk:
                yield final_chunk