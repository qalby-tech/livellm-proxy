#!/usr/bin/env python3
"""Direct WS test for asr-rust-ru — bypasses livellm-proxy.

Connects straight to the asr-rust-ru WebSocket endpoint, streams audio,
and prints every response so we can see exactly what the model returns
for known-good input. Use this to isolate whether transcription failures
live in the proxy (resampling, chunking) or in asr-rust-ru itself.

USAGE
─────
A. Inside the cluster (the proxy pod already has python + websockets):

    kubectl -n vodovoz cp test_asr_ws.py \
      api-proxy-livellm-proxy-86468b4d6d-spppc:/tmp/test_asr_ws.py
    kubectl -n vodovoz exec api-proxy-livellm-proxy-86468b4d6d-spppc -- \
      python3 /tmp/test_asr_ws.py /tmp/sample.wav

B. From your laptop via port-forward:

    kubectl -n vodovoz port-forward svc/gigaam-gigaam-rust 8081:8081 &
    ASR_WS_URL=ws://localhost:8081/v1/audio/transcriptions/ws?sample_rate=16000 \
      python3 test_asr_ws.py sample.wav

ARGS
────
    test_asr_ws.py                 — sends 2 s of generated tone (sanity check)
    test_asr_ws.py path/to.wav     — streams the WAV (must be 16 kHz mono PCM16)
    test_asr_ws.py path/to.wav 8000 — declare your WAV is 8 kHz; script upsamples to 16 kHz

ENV
───
    ASR_WS_URL  Override the WS URL (default = in-cluster gigaam DNS).
    CHUNK_MS    Chunk size in milliseconds (default 100).
"""

import asyncio
import audioop
import base64
import json
import math
import os
import struct
import sys
import wave

import websockets


URL = os.environ.get(
    "ASR_WS_URL",
    "ws://gigaam-gigaam-rust.vodovoz.svc.cluster.local:8081"
    "/v1/audio/transcriptions/ws?sample_rate=16000",
)
SAMPLE_RATE = 16000  # asr-rust-ru's only supported rate
CHUNK_MS = int(os.environ.get("CHUNK_MS", "100"))
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # PCM16


def generate_tone_pcm16(duration_s: float, freq_hz: float = 440.0, amp: float = 0.3) -> bytes:
    """Tone for sanity-checking the WS handshake. Should produce low speech_prob."""
    n = int(SAMPLE_RATE * duration_s)
    return struct.pack(
        f"<{n}h",
        *(int(amp * 32767 * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE)) for i in range(n)),
    )


def load_wav_pcm16(path: str, declared_sr: int = 0) -> bytes:
    """Load WAV and return PCM16 mono at SAMPLE_RATE. Upsamples if needed."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        pcm = w.readframes(n)
        print(f"WAV: rate={sr} channels={ch} sample_width={sw} frames={n} bytes={len(pcm)}")
        if sw != 2:
            raise ValueError(f"need 16-bit PCM, got {sw * 8}-bit")
        if ch != 1:
            # downmix to mono
            pcm = audioop.tomono(pcm, sw, 0.5, 0.5)
            print("  downmixed to mono")
        if declared_sr and declared_sr != sr:
            print(f"  treating as {declared_sr} Hz instead of {sr}")
            sr = declared_sr
        if sr != SAMPLE_RATE:
            pcm, _ = audioop.ratecv(pcm, 2, 1, sr, SAMPLE_RATE, None)
            print(f"  resampled {sr} → {SAMPLE_RATE} Hz | new bytes={len(pcm)}")
        return pcm


async def main():
    declared_sr = 0
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            declared_sr = int(sys.argv[2])
        pcm = load_wav_pcm16(sys.argv[1], declared_sr)
        print(f"Loaded {len(pcm)} bytes ({len(pcm) / 2 / SAMPLE_RATE:.2f} s of PCM16 mono @ {SAMPLE_RATE} Hz)")
    else:
        pcm = generate_tone_pcm16(2.0)
        print(f"Generated {len(pcm)} bytes of 440 Hz tone (sanity test — no real speech)")

    print(f"Connecting to {URL}")
    async with websockets.connect(URL, ping_interval=20, max_size=None) as ws:
        print("Connected. Streaming…")

        async def receiver():
            n = 0
            try:
                async for raw in ws:
                    n += 1
                    try:
                        ev = json.loads(raw)
                        text = ev.get("text", "")
                        prob = ev.get("speech_prob")
                        conf = ev.get("token_confidence")
                        samples = ev.get("samples")
                        etype = ev.get("type") or ev.get("event_type")
                        print(
                            f"[recv #{n}] type={etype!r} text={text!r} "
                            f"speech_prob={prob} token_conf={conf} samples={samples}"
                        )
                    except json.JSONDecodeError:
                        print(f"[recv #{n}] non-JSON: {raw!r}")
            except websockets.ConnectionClosed as e:
                print(f"[recv] WS closed: code={e.code} reason={e.reason!r}")
            print(f"[recv] done — {n} messages total")

        recv_task = asyncio.create_task(receiver())

        sent = 0
        for i in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[i:i + CHUNK_BYTES]
            await ws.send(json.dumps({
                "audio": base64.b64encode(chunk).decode("ascii"),
                "final": False,
            }))
            sent += 1
            if sent == 1 or sent % 10 == 0:
                print(f"[send] chunk #{sent} bytes={len(chunk)}")
            await asyncio.sleep(CHUNK_MS / 1000)

        # Force-flush the segment buffer
        await ws.send(json.dumps({"audio": "", "final": True}))
        print(f"[send] sent {sent} chunks + final. Waiting 2 s for late deltas…")

        await asyncio.sleep(2.0)
        await ws.close()
        await recv_task


if __name__ == "__main__":
    asyncio.run(main())
