"""
Unit tests for config-driven audio fallback in AudioManager.

These exercise AudioManager.safe_speak / safe_transcribe against a real
FallbackManager, verifying that a `fallback` configured on the provider's
ModelConfig is honored server-side (mirroring the agent path) — without the
client having to send an explicit *FallbackRequest.
"""

import pytest
from unittest.mock import MagicMock

from managers.audio import AudioManager
from managers.fallback import FallbackManager
from managers.config import ConfigManager
from models.common import ModelConfig, FallbackConfig, FallbackStrategyType
from models.audio.speak import SpeakRequest, SpeakResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse


PRIMARY_UID = "vodovoz-zvonilka-vox-provider"
FALLBACK_UID = "vodovoz-zvonilka-f5-provider"


def _speak_payload() -> SpeakRequest:
    return SpeakRequest(
        provider_uid=PRIMARY_UID,
        model="tts-1",
        text="привет мир",
        voice="alloy",
        mime_type="audio/pcm",
        sample_rate=24000,
    )


def _manager_with_fallback(model_config):
    """AudioManager whose ConfigManager returns `model_config` for any model."""
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_model_config.return_value = model_config
    return AudioManager(config_manager=config_manager, fallback_manager=FallbackManager())


@pytest.mark.audio
@pytest.mark.asyncio
async def test_safe_speak_falls_back_to_f5_when_vox_fails(monkeypatch):
    """A plain SpeakRequest fails on vox → automatically retries on the f5 provider."""
    model_config = ModelConfig(
        fallback=FallbackConfig(
            fallback_provider_uid=FALLBACK_UID,
            fallback_model="tts-1",
            fallback_strategy=FallbackStrategyType.SEQUENTIAL,
        )
    )
    manager = _manager_with_fallback(model_config)

    calls = []

    async def fake_speak(request: SpeakRequest, stream: bool = False):
        calls.append(request.provider_uid)
        if request.provider_uid == PRIMARY_UID:
            raise RuntimeError("vox is down")
        return SpeakResponse(audio=b"f5-audio", content_type="audio/pcm", sample_rate=24000)

    monkeypatch.setattr(manager, "speak", fake_speak)

    result = await manager.safe_speak(_speak_payload())

    assert result.audio == b"f5-audio"
    # primary tried first, then fallback — text/voice/format preserved on the copy
    assert calls == [PRIMARY_UID, FALLBACK_UID]


@pytest.mark.audio
@pytest.mark.asyncio
async def test_safe_speak_no_fallback_when_unconfigured(monkeypatch):
    """Without a configured fallback, a failing SpeakRequest is not retried."""
    manager = _manager_with_fallback(None)

    calls = []

    async def fake_speak(request: SpeakRequest, stream: bool = False):
        calls.append(request.provider_uid)
        raise RuntimeError("vox is down")

    monkeypatch.setattr(manager, "speak", fake_speak)

    with pytest.raises(RuntimeError):
        await manager.safe_speak(_speak_payload())

    assert calls == [PRIMARY_UID]  # no fallback attempted


@pytest.mark.audio
@pytest.mark.asyncio
async def test_safe_speak_primary_success_skips_fallback(monkeypatch):
    """When the primary succeeds, the fallback provider is never touched."""
    model_config = ModelConfig(
        fallback=FallbackConfig(
            fallback_provider_uid=FALLBACK_UID,
            fallback_model="tts-1",
        )
    )
    manager = _manager_with_fallback(model_config)

    calls = []

    async def fake_speak(request: SpeakRequest, stream: bool = False):
        calls.append(request.provider_uid)
        return SpeakResponse(audio=b"vox-audio", content_type="audio/pcm", sample_rate=24000)

    monkeypatch.setattr(manager, "speak", fake_speak)

    result = await manager.safe_speak(_speak_payload())

    assert result.audio == b"vox-audio"
    assert calls == [PRIMARY_UID]


@pytest.mark.audio
@pytest.mark.asyncio
async def test_safe_transcribe_falls_back_when_primary_fails(monkeypatch):
    """Config-driven fallback applies to transcription too."""
    model_config = ModelConfig(
        fallback=FallbackConfig(
            fallback_provider_uid="fallback-stt",
            fallback_model="whisper-1",
        )
    )
    manager = _manager_with_fallback(model_config)

    calls = []

    async def fake_transcribe(request: TranscribeRequest):
        calls.append(request.provider_uid)
        if request.provider_uid == "primary-stt":
            raise RuntimeError("primary stt down")
        return TranscribeResponse(text="hello", language="en")

    monkeypatch.setattr(manager, "transcribe", fake_transcribe)

    payload = TranscribeRequest(
        provider_uid="primary-stt",
        model="whisper-1",
        file=("a.wav", b"audio-bytes", "audio/wav"),
        language="en",
    )
    result = await manager.safe_transcribe(payload)

    assert result.text == "hello"
    assert calls == ["primary-stt", "fallback-stt"]
