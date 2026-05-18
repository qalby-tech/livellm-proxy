"""Livellm realtime transcription service.

Connects to a self-hosted livellm-compatible WebSocket ASR endpoint (the one
shipped by `asr-rust-ru` at `/v1/audio/transcriptions/ws`) and turns the raw
delta stream into VAD-aware transcription chunks.

The VAD is driven by the upstream ASR's `speech_prob` signal (`1 − P(blank)`
averaged over encoder frames) so there's no separate VAD model — the ASR
itself tells us when speech is present.
"""

import asyncio
import base64
import json
import logging
import time
from contextlib import suppress
from typing import AsyncIterator, Awaitable, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from audio_ai.base import AudioRealtimeTranscriptionService
from managers import telemetry as logfire
from models.audio.speak import SpeakMimeType
from models.audio.transcription_ws import TranscriptionWsResponse


# Default tunables — every one of them can be overridden via `gen_config`.
DEFAULTS = {
    # VAD thresholds applied to `speech_prob` (0..1).
    "speech_start_threshold": 0.55,
    "speech_end_threshold": 0.30,
    # Consecutive chunks needed to confirm a start/end transition. Prevents
    # single-chunk blips from opening/closing segments.
    "start_hold_chunks": 2,
    "end_hold_chunks": 4,
    # Drop a delta if its token-confidence falls below this value. Useful for
    # filtering out hallucinations on noise. Set to 0 to disable.
    "min_token_confidence": 0.0,
    # Only emit a delta if the new text differs from the previously emitted one.
    "dedupe": True,
    # If True, suppress deltas while the VAD considers us in silence.
    "suppress_silence": True,
    # Hard ceiling on a single utterance (seconds). When exceeded, the buffer
    # is force-committed so downstream consumers don't starve.
    "max_utterance_seconds": 30.0,
}


class LivellmRealtimeTranscriptionService(AudioRealtimeTranscriptionService):
    """Realtime transcription against a livellm-compatible WebSocket endpoint.

    Wire protocol (mirror of `asr-rust-ru`):

        client → server   {"audio": "<base64 pcm16>", "final": false}
        server → client   {"type": "delta", "text": "...",
                           "token_confidence": 0.91, "speech_prob": 0.83,
                           "samples": 12800}

    `final: true` flushes the server buffer for the current segment.

    The base class is responsible for resampling the inbound audio to
    `default_sample_rate` (16 kHz here) and decoding ulaw/alaw/PCM into raw
    PCM16 before calling `send_audio_chunk`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "gigaam",
        language: str = "auto",
        gen_config: Optional[dict] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required for livellm provider")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._language = language

        cfg = {**DEFAULTS, **(gen_config or {})}
        self._start_threshold: float = float(cfg["speech_start_threshold"])
        self._end_threshold: float = float(cfg["speech_end_threshold"])
        self._start_hold: int = int(cfg["start_hold_chunks"])
        self._end_hold: int = int(cfg["end_hold_chunks"])
        self._min_token_conf: float = float(cfg["min_token_confidence"])
        self._dedupe: bool = bool(cfg["dedupe"])
        self._suppress_silence: bool = bool(cfg["suppress_silence"])
        self._max_utterance_s: float = float(cfg["max_utterance_seconds"])

        # Connection state.
        self._ws: Optional[ClientConnection] = None
        # VAD state machine.
        self._in_speech = False
        self._start_streak = 0
        self._end_streak = 0
        self._last_emitted_text = ""
        self._segment_start_ts: Optional[float] = None
        # Set by the receiver when VAD detects end-of-speech; the sender
        # consumes it on its next chunk to flush asr-rust-ru's accumulator
        # so each turn is transcribed independently (otherwise the buffer
        # grows for the whole call and every chunk re-transcribes the lot).
        self._pending_final: bool = False

    @property
    def default_sample_rate(self) -> int:
        # asr-rust-ru only supports 16 kHz. The base class resamples on send.
        return 16000

    def _build_ws_url(self) -> str:
        if self._base_url.startswith("http://"):
            scheme = "ws://"
            host = self._base_url[len("http://") :]
        elif self._base_url.startswith("https://"):
            scheme = "wss://"
            host = self._base_url[len("https://") :]
        elif self._base_url.startswith("ws://") or self._base_url.startswith("wss://"):
            return f"{self._base_url}/v1/audio/transcriptions/ws?sample_rate={self.default_sample_rate}"
        else:
            scheme = "ws://"
            host = self._base_url
        return f"{scheme}{host}/v1/audio/transcriptions/ws?sample_rate={self.default_sample_rate}"

    async def connect(self) -> None:
        url = self._build_ws_url()
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        logfire.info(f"livellm: connecting to {url}")
        self._ws = await websockets.connect(
            url,
            additional_headers=headers or None,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

    async def disconnect(self) -> None:
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def send_audio_chunk(self, audio_source: AsyncIterator[bytes]) -> None:
        """Forward PCM16/16 kHz audio chunks to the upstream as base64 frames.

        We also keep a clock so we can emit a synthetic `final: true` if the
        speaker has been talking for longer than `max_utterance_seconds`.
        """
        if self._ws is None:
            raise RuntimeError("send_audio_chunk called before connect()")

        async for chunk in audio_source:
            if not chunk:
                continue
            # Decide whether to ask the server to flush its buffer.
            do_final = False
            if self._pending_final:
                # VAD detected end-of-speech on the receive side.
                do_final = True
                self._pending_final = False
            elif self._segment_start_ts is not None and self._max_utterance_s > 0:
                # Hard-cap fallback: flush a single utterance that has run
                # past max_utterance_seconds without the VAD calling it.
                if time.monotonic() - self._segment_start_ts > self._max_utterance_s:
                    do_final = True
                    self._segment_start_ts = None
            payload = {
                "audio": base64.b64encode(chunk).decode("ascii"),
                "final": do_final,
            }
            try:
                await self._ws.send(json.dumps(payload))
            except websockets.ConnectionClosed:
                logfire.info("livellm: upstream closed during send")
                return

    async def receive_audio_chunk(
        self, audio_sink: Callable[[str], Awaitable[None]]
    ) -> None:
        """Consume upstream deltas, apply VAD + filters, forward survivors.

        VAD strategy: walk the `speech_prob` value reported per chunk.
        * Cross above `speech_start_threshold` for `start_hold_chunks` → speech starts.
        * Cross below `speech_end_threshold` for `end_hold_chunks` → speech ends
          and we ask the server to flush the buffer (sent on the next inbound
          audio chunk via the `final` flag).
        """
        if self._ws is None:
            raise RuntimeError("receive_audio_chunk called before connect()")

        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    logfire.warn(f"livellm: dropping non-JSON frame: {raw!r}")
                    continue

                etype = event.get("type")
                if etype == "error":
                    logfire.error(f"livellm: upstream error: {event.get('error')}")
                    continue
                if etype not in ("delta", "final"):
                    continue

                speech_prob = float(event.get("speech_prob", 0.0))
                token_conf = float(event.get("token_confidence", 0.0))
                text = event.get("text", "") or ""

                # ─── VAD state machine ──────────────────────────────────
                if speech_prob >= self._start_threshold:
                    self._start_streak += 1
                    self._end_streak = 0
                else:
                    self._start_streak = 0
                if speech_prob <= self._end_threshold:
                    self._end_streak += 1
                else:
                    self._end_streak = 0

                if not self._in_speech and self._start_streak >= self._start_hold:
                    self._in_speech = True
                    self._segment_start_ts = time.monotonic()
                    logfire.info(
                        f"livellm: speech START (speech_prob={speech_prob:.2f})"
                    )

                ended = False
                if self._in_speech and self._end_streak >= self._end_hold:
                    ended = True
                    self._in_speech = False
                    self._start_streak = 0
                    self._segment_start_ts = None
                    # Ask the sender to flush asr-rust-ru's buffer on its
                    # next outbound chunk so the next turn starts clean.
                    self._pending_final = True
                    logfire.info(
                        f"livellm: speech END (speech_prob={speech_prob:.2f}) — requesting flush"
                    )

                # ─── Filtering ──────────────────────────────────────────
                if self._suppress_silence and not self._in_speech and not ended:
                    continue
                if token_conf < self._min_token_conf and not ended:
                    continue
                if not text and not ended:
                    continue
                if self._dedupe and text == self._last_emitted_text and not ended:
                    continue

                self._last_emitted_text = text
                await audio_sink(text)
        except websockets.ConnectionClosed:
            logfire.info("livellm: upstream closed during receive")
