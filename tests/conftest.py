"""Pytest configuration and fixtures for testing the FastAPI application."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

# ── Env vars — must be set before the app module is imported ─────────────────
# redis_url is a required field in EnvSettings; pydantic-settings reads it from
# the environment at import time, so it must already be present here.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


# ── Redis mocks ───────────────────────────────────────────────────────────────
# Tests must never depend on a running Redis server.  We patch three methods on
# the manager classes *before* `from main import app` so the patches are already
# active when the lifespan context manager runs inside TestClient.


def _make_mock_redis_client() -> MagicMock:
    """Return a MagicMock that satisfies every Redis call the app makes."""
    client = MagicMock()
    client.hgetall = AsyncMock(return_value={})  # load_all_provider_settings
    client.hget = AsyncMock(return_value=None)  # load_provider_settings
    client.hset = AsyncMock(return_value=1)  # save_provider_settings
    client.hdel = AsyncMock(return_value=1)  # delete_provider_settings
    client.publish = AsyncMock(return_value=0)  # publish_provider_event
    return client


async def _mock_connect(self) -> None:
    """Replace RedisManager.connect — wires up a mock client, no network I/O."""
    self.redis_client = _make_mock_redis_client()


async def _mock_disconnect(self) -> None:
    """Replace RedisManager.disconnect — no-op in tests."""


async def _mock_pubsub_listener(self) -> None:
    """Replace ConfigManager.pubsub_listener_task — sleeps until cancelled."""
    try:
        await asyncio.sleep(float("inf"))
    except asyncio.CancelledError:
        pass


patch("managers.redis.RedisManager.connect", _mock_connect).start()
patch("managers.redis.RedisManager.disconnect", _mock_disconnect).start()
patch(
    "managers.config.ConfigManager.pubsub_listener_task", _mock_pubsub_listener
).start()

# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import AsyncMock, MagicMock  # re-exported for fixtures below

import pytest
from fastapi.testclient import TestClient

from main import app
from managers.agent import AgentManager
from managers.audio import AudioManager
from routers.agent import get_agent_manager
from routers.audio import get_audio_manager

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_agent_manager():
    """Mock AgentManager with an awaitable run method."""
    manager = MagicMock(spec=AgentManager)
    manager.run = AsyncMock()
    return manager


@pytest.fixture
def mock_audio_manager():
    """Mock AudioManager with awaitable speak and transcribe methods."""
    manager = MagicMock(spec=AudioManager)
    manager.speak = AsyncMock()
    manager.transcribe = AsyncMock()
    return manager


@pytest.fixture
def client(mock_agent_manager, mock_audio_manager):
    """
    TestClient with dependency overrides for AgentManager and AudioManager.

    The lifespan runs inside the `with` block:
      - RedisManager.connect   → _mock_connect   (mock client, no network)
      - pubsub_listener_task   → _mock_pubsub_listener (cancelled on teardown)
    """
    app.dependency_overrides[get_agent_manager] = lambda: mock_agent_manager
    app.dependency_overrides[get_audio_manager] = lambda: mock_audio_manager

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ── Request payload factories ─────────────────────────────────────────────────


@pytest.fixture
def agent_payload():
    return {
        "provider_uid": "test-provider-uid",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "tools": [],
    }


@pytest.fixture
def audio_speak_payload():
    return {
        "provider_uid": "test-provider-uid",
        "model": "tts-1",
        "text": "Hello world",
        "voice": "alloy",
        "mime_type": "audio/pcm",
        "sample_rate": 24000,
    }


@pytest.fixture
def audio_transcribe_payload():
    return {
        "provider_uid": "test-provider-uid",
        "model": "whisper-1",
        "file": ("test-audio.wav", b"test-audio-content", "audio/pcm"),
        "language": "en",
    }
