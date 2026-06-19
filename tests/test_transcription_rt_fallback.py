"""
Unit tests for connect-time fallback in TranscriptionRTManager (realtime ASR WS).

Realtime transcription is a long-lived WebSocket, so fallback is applied at
connect time: if the primary provider can't connect, the configured fallback
provider/model is connected instead — driven by the same ModelConfig.fallback
that powers the HTTP audio / agent paths.
"""

import pytest
from unittest.mock import MagicMock

from managers.transcription_rt import TranscriptionRTManager
from managers.config import ConfigManager
from models.common import ModelConfig, FallbackConfig
from models.audio.transcription_ws import TranscriptionInitWsRequest


PRIMARY_UID = "vodovoz-giga-am-rust-provider"
FALLBACK_UID = "vodovoz-whisper-openai-provider"


class _FakeService:
    """Stand-in for an AudioRealtimeTranscriptionService."""

    def __init__(self, uid, model, fail_connect=False):
        self.uid = uid
        self.model = model
        self.fail_connect = fail_connect
        self.connected = False
        self.disconnected = False

    async def connect(self):
        if self.fail_connect:
            raise ConnectionError(f"cannot reach {self.uid}")
        self.connected = True

    async def disconnect(self):
        self.disconnected = True


def _init_request(uid=PRIMARY_UID, model="whisper-1"):
    return TranscriptionInitWsRequest(provider_uid=uid, model=model, language="ru")


def _manager(model_config):
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_model_config.return_value = model_config
    return TranscriptionRTManager(config_manager=config_manager)


@pytest.mark.audio
@pytest.mark.asyncio
async def test_connect_falls_back_when_primary_unreachable(monkeypatch):
    model_config = ModelConfig(
        fallback=FallbackConfig(fallback_provider_uid=FALLBACK_UID, fallback_model="whisper-1")
    )
    manager = _manager(model_config)

    built = {}

    def fake_build(provider_uid, model, request):
        svc = _FakeService(provider_uid, model, fail_connect=(provider_uid == PRIMARY_UID))
        built[provider_uid] = svc
        return svc

    monkeypatch.setattr(manager, "_build_service", fake_build)

    service = await manager.create_connected_service(_init_request())

    assert service.uid == FALLBACK_UID
    assert service.connected is True
    # the failed primary service was cleaned up
    assert built[PRIMARY_UID].disconnected is True


@pytest.mark.audio
@pytest.mark.asyncio
async def test_connect_uses_primary_when_available(monkeypatch):
    model_config = ModelConfig(
        fallback=FallbackConfig(fallback_provider_uid=FALLBACK_UID, fallback_model="whisper-1")
    )
    manager = _manager(model_config)

    def fake_build(provider_uid, model, request):
        return _FakeService(provider_uid, model, fail_connect=False)

    monkeypatch.setattr(manager, "_build_service", fake_build)

    service = await manager.create_connected_service(_init_request())

    assert service.uid == PRIMARY_UID
    assert service.connected is True


@pytest.mark.audio
@pytest.mark.asyncio
async def test_connect_raises_when_all_targets_fail(monkeypatch):
    model_config = ModelConfig(
        fallback=FallbackConfig(fallback_provider_uid=FALLBACK_UID, fallback_model="whisper-1")
    )
    manager = _manager(model_config)

    def fake_build(provider_uid, model, request):
        return _FakeService(provider_uid, model, fail_connect=True)

    monkeypatch.setattr(manager, "_build_service", fake_build)

    with pytest.raises(RuntimeError):
        await manager.create_connected_service(_init_request())


@pytest.mark.audio
@pytest.mark.asyncio
async def test_connect_no_fallback_when_unconfigured(monkeypatch):
    manager = _manager(None)  # no model config → no fallback target

    def fake_build(provider_uid, model, request):
        return _FakeService(provider_uid, model, fail_connect=True)

    monkeypatch.setattr(manager, "_build_service", fake_build)

    with pytest.raises(RuntimeError):
        await manager.create_connected_service(_init_request())

    # only the primary was attempted (no fallback configured)
    manager.config_manager.get_model_config.assert_called_once()
